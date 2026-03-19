from typing import List, Optional
import json
from datetime import datetime
import logging

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File, APIRouter, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from artifacts import artifact_diagnostics, load_manifest
from feature_spec import FEATURE_COLUMNS, ARIMAX_EXOG_COLUMNS, SEQUENCE_LENGTH
from evaluation_service import compare_models
from database import get_db, init_db, engine
from models import User, PatientFlow, MessageLog, OptimizationRun
from resource_optimizer import optimize_resources
from schemas import LoginRequest
from etl_pipeline import ingest_patient_flow, ingest_appointments, ingest_or
from auth import create_token, bearer_from_header, decode_token, verify_password
from forecast_inference import load_assets as _load_assets, predict_hybrid as _predict_hybrid

app = FastAPI(title="Hospital AI API")
logging.basicConfig(level=logging.INFO)

# Routers (keep public URLs stable; we can version later)
system_router = APIRouter(tags=["system"])
auth_router = APIRouter(prefix="/auth", tags=["auth"])
messages_router = APIRouter(prefix="/messages", tags=["messages"])
patient_flow_router = APIRouter(prefix="/patient_flow", tags=["patient_flow"])
ml_router = APIRouter(tags=["ml"])
upload_router = APIRouter(prefix="/upload", tags=["upload"])


@app.on_event("startup")
def _startup_create_tables():
    # Safe default for this repo: ensure tables exist at runtime.
    # For production, migrate to Alembic migrations and remove create_all.
    init_db()

LEGACY_MESSAGES_FILE = "messages_log.csv"  # import-only (not runtime)
LEGACY_MESSAGE_COLS = [
    "message_id",
    "timestamp",
    "sender_role",
    "sender_name",
    "target_role",
    "target_department",
    "priority",
    "category",
    "title",
    "message",
    "status",
    "reply",
    "reply_by",
    "reply_timestamp",
    "acknowledged",
    "archived",
]

FEATURE_COUNT = len(FEATURE_COLUMNS)


def _get_assets_or_503():
    try:
        return _load_assets()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Forecasting artifacts not ready: {e}")

ADMIN_MESSAGE_TEMPLATES = [
    {
        "category": "emergency",
        "priority": "critical",
        "title": "Emergency Surge Alert",
        "message": "Emergency surge alert: all available staff should review current assignments and prepare for overflow response.",
        "target_role": "all",
        "target_department": "All Departments",
    },
    {
        "category": "coverage",
        "priority": "high",
        "title": "Doctor Coverage Request",
        "message": "Urgent coverage needed: an additional doctor is required to cover the current shift immediately.",
        "target_role": "doctor",
        "target_department": "All Departments",
    },
    {
        "category": "coverage",
        "priority": "high",
        "title": "Nurse Coverage Request",
        "message": "Urgent coverage needed: an additional nurse is required to support the active department.",
        "target_role": "nurse",
        "target_department": "All Departments",
    },
    {
        "category": "shift",
        "priority": "high",
        "title": "Shift Change Notice",
        "message": "Shift update notice: please review your latest assignment and acknowledge the change.",
        "target_role": "all",
        "target_department": "All Departments",
    },
    {
        "category": "capacity",
        "priority": "high",
        "title": "Bed Shortage Warning",
        "message": "Capacity warning: bed pressure is increasing. Review admissions and discharge flow immediately.",
        "target_role": "all",
        "target_department": "All Departments",
    },
]

STAFF_QUICK_REPLIES = [
    "تم",
    "تم التنفيذ",
    "وصلت",
    "جاري التنفيذ",
    "نحتاج دعم دكاترة",
    "نحتاج دعم تمريض",
    "يوجد عجز",
    "لا أستطيع التغطية الآن",
]


class PredictRequest(BaseModel):
    sequence: List[List[float]]


class SimulateRequest(BaseModel):
    predicted_patients: float
    beds_available: int
    doctors_available: int
    demand_increase_percent: float = 0


class ExplainRequest(BaseModel):
    sequence: List[List[float]]


class SendMessageRequest(BaseModel):
    sender_role: str
    sender_name: str
    target_role: str = "all"
    target_department: str = "All Departments"
    priority: str = "normal"
    category: str = "general"
    title: str
    message: str


class ReplyMessageRequest(BaseModel):
    message_id: str
    reply: str
    reply_by: str


class MessageActionRequest(BaseModel):
    message_id: str


class EvaluateRequest(BaseModel):
    actual: List[float]
    lstm: List[float]
    arimax: List[float]
    hybrid: List[float]


def get_token_payload(authorization: Optional[str] = Header(default=None)) -> dict:
    """Extract and decode JWT from Authorization: Bearer ..."""

    token = bearer_from_header(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    try:
        return decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_role(roles: List[str]):
    allowed = {r.lower() for r in roles}

    def _dep(payload: dict = Depends(get_token_payload)) -> dict:
        role = str(payload.get("role", "")).lower()
        if role not in allowed:
            raise HTTPException(status_code=403, detail=f"Forbidden for role={role}")
        return payload

    return _dep


require_admin = require_role(["admin"])
require_staff_or_admin = require_role(["admin", "doctor", "nurse"])


def build_engineered_sequence_from_patient_flow(rows: List[PatientFlow]) -> List[List[float]]:
    """Build engineered sequence from DB rows.

    NOTE: This uses the same deterministic feature engineering from Phase 1.
    """

    from forecast_features import build_latest_sequence_from_rows

    payload_rows = [
        {
            "patients": float(r.patients),
            "day_of_week": float(r.day_of_week or 0),
            "month": float(r.month or 0),
            "is_weekend": float(r.is_weekend or 0),
            "holiday": float(r.holiday or 0),
            "weather": float(r.weather or 0),
            "datetime": getattr(r, "datetime", None),
        }
        for r in rows
    ]

    return build_latest_sequence_from_rows(payload_rows)


def normalize_text(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    if text.lower() == "nan":
        return default
    return text if text else default


def normalize_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in ["true", "1", "yes", "y"]:
        return True
    if text in ["false", "0", "no", "n", ""]:
        return False
    return default


def parse_datetime_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


def validate_sequence_shape(arr: np.ndarray):
    return arr.shape == (SEQUENCE_LENGTH, FEATURE_COUNT)


def predict_hybrid(sequence_array: np.ndarray):
    # Delegate to canonical inference module.
    return _predict_hybrid(sequence_array)


def predict_emergency_load(predicted_patients: float):
    if predicted_patients < 80:
        return "LOW"
    if predicted_patients < 120:
        return "MEDIUM"
    return "HIGH"


def allocate_beds(predicted_patients: int, available_beds: int):
    if predicted_patients <= available_beds:
        return {
            "status": "OK",
            "beds_used": predicted_patients,
            "beds_remaining": available_beds - predicted_patients,
            "shortage": 0,
        }

    shortage = predicted_patients - available_beds
    return {
        "status": "SHORTAGE",
        "beds_used": available_beds,
        "beds_remaining": 0,
        "shortage": shortage,
    }


def calculate_recommended_resources(predicted_patients: float):
    return {
        "beds_needed": int(np.ceil(predicted_patients * 1.15)),
        "doctors_needed": max(1, int(np.ceil(predicted_patients / 6.0))),
        "nurses_needed": max(1, int(np.ceil(predicted_patients / 3.5))),
    }


def explain_feature_importance(sequence_array: np.ndarray):
    base_result = predict_hybrid(sequence_array)
    base_pred = float(base_result["hybrid_prediction"])
    impacts = []

    for i, feature_name in enumerate(FEATURE_COLUMNS):
        modified = sequence_array.copy()

        if feature_name == "patients":
            modified[-1, i] = modified[-1, i] * 1.10
        elif feature_name in ["day_of_week", "month", "weather", "hour", "trend_feature"]:
            modified[-1, i] = modified[-1, i] + 1
        elif feature_name in ["is_weekend", "holiday"]:
            modified[-1, i] = 1 - modified[-1, i]
        else:
            modified[-1, i] = modified[-1, i] * 1.05

        new_pred = float(predict_hybrid(modified)["hybrid_prediction"])
        impacts.append({"feature": feature_name, "impact": float(new_pred - base_pred)})

    impacts = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)
    return {"base_prediction": float(base_pred), "feature_impacts": impacts}


# (duplicate removed)


def serialize_message_row(row: MessageLog) -> dict:
    return {
        "message_id": normalize_text(row.message_id),
        "timestamp": normalize_text(row.timestamp),
        "sender_role": normalize_text(row.sender_role),
        "sender_name": normalize_text(row.sender_name),
        "target_role": normalize_text(row.target_role, "all"),
        "target_department": normalize_text(row.target_department, "All Departments"),
        "priority": normalize_text(row.priority, "normal"),
        "category": normalize_text(row.category, "general"),
        "title": normalize_text(row.title),
        "message": normalize_text(row.message),
        "status": normalize_text(row.status, "sent"),
        "reply": normalize_text(row.reply),
        "reply_by": normalize_text(row.reply_by),
        "reply_timestamp": normalize_text(row.reply_timestamp),
        "acknowledged": normalize_text(row.acknowledged, "no"),
        "archived": bool(row.archived),
    }


def bootstrap_messages_from_csv_if_needed(db: Session):
    # DB-first runtime: do not bootstrap from CSV.
    return


@app.middleware("http")
async def log_requests(request, call_next):
    logging.info("%s %s", request.method, request.url)
    response = await call_next(request)
    return response


@system_router.get("/")
def home(_token: dict = Depends(require_staff_or_admin)):
    return {"message": "Hospital AI API is running"}


@system_router.get("/health")
def health(_token: dict = Depends(require_staff_or_admin)):
    return {"status": "ok"}


@system_router.get("/health/db")
def health_db(_token: dict = Depends(require_staff_or_admin)):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"db_unhealthy: {e}")


@system_router.get("/status")
def system_status(_token: dict = Depends(require_staff_or_admin)):
    diag = artifact_diagnostics()
    manifest = load_manifest()
    weights = None

    if not diag.get("missing"):
        try:
            assets = _get_assets_or_503()
            weights = {
                "lstm": float(getattr(assets, "lstm_weight", None)),
                "arimax": float(getattr(assets, "arimax_weight", None)),
            }
        except Exception:
            weights = None

    return {
        "system": "Hospital AI",
        "model": "Hybrid Forecast (LSTM + ARIMAX)",
        "status": "running",
        "hybrid_weights": weights or {"lstm": None, "arimax": None},
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "artifacts": diag,
        "artifact_manifest": manifest,
    }


@system_router.get("/feature_config")
def get_feature_config(_token: dict = Depends(require_staff_or_admin)):
    return {
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "arimax_exog_columns": ARIMAX_EXOG_COLUMNS,
    }


@system_router.get("/artifacts/manifest")
def get_artifacts_manifest(_token: dict = Depends(require_admin)):
    return load_manifest()


@messages_router.get("/templates")
def get_message_templates(_token: dict = Depends(require_staff_or_admin)):
    return {
        "admin_templates": ADMIN_MESSAGE_TEMPLATES,
        "staff_quick_replies": STAFF_QUICK_REPLIES,
    }


@messages_router.get("")
def get_messages(
    role: Optional[str] = Query(default=None),
    department: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    unread_only: bool = Query(default=False),
    include_archived: bool = Query(default=False),
    sender_name: Optional[str] = Query(default=None),
    _token: dict = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    bootstrap_messages_from_csv_if_needed(db)
    query = db.query(MessageLog)

    if role:
        role = normalize_text(role).lower()
        query = query.filter(
            (MessageLog.target_role.ilike(role))
            | (MessageLog.target_role.ilike("all"))
        )

    if department:
        department = normalize_text(department).lower()
        query = query.filter(
            (MessageLog.target_department.ilike(department))
            | (MessageLog.target_department.ilike("all departments"))
            | (MessageLog.target_department.ilike("all"))
        )

    if sender_name:
        query = query.filter(MessageLog.sender_name.ilike(normalize_text(sender_name)))

    if unread_only:
        query = query.filter(MessageLog.acknowledged.ilike("no"))

    if include_archived:
        query = query.filter(MessageLog.archived.is_(True))
    else:
        query = query.filter(
            (MessageLog.archived.is_(False)) | (MessageLog.archived.is_(None))
        )

    rows = query.order_by(MessageLog.id.desc()).limit(limit).all()
    return {
        "messages": [serialize_message_row(row) for row in rows],
        "quick_replies": STAFF_QUICK_REPLIES,
    }


@messages_router.post("/send")
def send_message(
    payload: SendMessageRequest,
    _token: dict = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    bootstrap_messages_from_csv_if_needed(db)

    row = MessageLog(
        message_id=f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        timestamp=parse_datetime_now(),
        sender_role=normalize_text(payload.sender_role, "admin"),
        sender_name=normalize_text(payload.sender_name, "Unknown Sender"),
        target_role=normalize_text(payload.target_role, "all"),
        target_department=normalize_text(payload.target_department, "All Departments"),
        priority=normalize_text(payload.priority, "normal"),
        category=normalize_text(payload.category, "general"),
        title=normalize_text(payload.title, "Untitled Message"),
        message=normalize_text(payload.message),
        status="sent",
        reply="",
        reply_by="",
        reply_timestamp="",
        acknowledged="no",
        archived=False,
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "status": "sent",
        "message": "Message sent successfully.",
        "data": serialize_message_row(row),
    }


@messages_router.post("/reply")
def reply_to_message(
    payload: ReplyMessageRequest,
    _token: dict = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    bootstrap_messages_from_csv_if_needed(db)

    row = db.query(MessageLog).filter(
        MessageLog.message_id == normalize_text(payload.message_id)
    ).first()

    if row is None:
        raise HTTPException(status_code=404, detail="Message not found.")

    row.reply = normalize_text(payload.reply)
    row.reply_by = normalize_text(payload.reply_by)
    row.reply_timestamp = parse_datetime_now()
    row.status = "updated"
    row.acknowledged = "yes"
    # Same rationale as ack: once actioned (replied), remove from inbox.
    row.archived = True

    db.commit()
    db.refresh(row)

    return {
        "status": "updated",
        "message": "Reply saved successfully.",
        "data": serialize_message_row(row),
    }


@messages_router.post("/ack")
def acknowledge_message(
    payload: MessageActionRequest,
    _token: dict = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    bootstrap_messages_from_csv_if_needed(db)

    row = db.query(MessageLog).filter(
        MessageLog.message_id == normalize_text(payload.message_id)
    ).first()

    if row is None:
        raise HTTPException(status_code=404, detail="Message not found.")

    # Product behavior: once staff acknowledges a message, move it out of inbox to reduce distraction.
    # Users can still view it under archive/history.
    row.acknowledged = "yes"
    row.archived = True
    db.commit()
    db.refresh(row)

    return {"status": "acknowledged", "data": serialize_message_row(row)}


@messages_router.post("/archive")
def archive_message(
    payload: MessageActionRequest,
    _token: dict = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    bootstrap_messages_from_csv_if_needed(db)

    row = db.query(MessageLog).filter(
        MessageLog.message_id == normalize_text(payload.message_id)
    ).first()

    if row is None:
        raise HTTPException(status_code=404, detail="Message not found.")

    row.archived = True
    row.acknowledged = "yes"
    db.commit()
    db.refresh(row)

    return {"status": "archived", "data": serialize_message_row(row)}


@auth_router.post("/login")
def login_user(payload: LoginRequest, db: Session = Depends(get_db)):
    username = normalize_text(payload.username)
    password = normalize_text(payload.password)

    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, normalize_text(user.password)):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_token(
        {
            "sub": normalize_text(user.username),
            "username": normalize_text(user.username),
            "role": normalize_text(user.role),
            "department": normalize_text(user.department),
            "name": normalize_text(user.name),
        }
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": normalize_text(user.username),
            "name": normalize_text(user.name),
            "role": normalize_text(user.role),
            "department": normalize_text(user.department),
        },
    }


@auth_router.get("/users")
def get_users(_token: dict = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {
        "users": [
            {
                "username": normalize_text(u.username),
                "name": normalize_text(u.name),
                "role": normalize_text(u.role),
                "department": normalize_text(u.department),
            }
            for u in users
        ]
    }


@patient_flow_router.get("/latest")
def get_latest_patient_flow_sequence(
    _token: dict = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    rows = db.query(PatientFlow).order_by(PatientFlow.id.desc()).limit(SEQUENCE_LENGTH).all()
    if len(rows) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {SEQUENCE_LENGTH} patient flow rows in DB.",
        )

    rows = list(reversed(rows))
    return {
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": FEATURE_COUNT,
        "sequence": build_engineered_sequence_from_patient_flow(rows),
    }


@ml_router.get("/optimize_resources/{predicted_patients}")
def optimize_resources_endpoint(
    predicted_patients: float,
    _token: dict = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    """Run optimization and persist the run for audit + approvals."""

    result = optimize_resources(predicted_patients)
    summary = result.get("summary", {}) if isinstance(result, dict) else {}

    try:
        row = OptimizationRun(
            run_id=_new_run_id("OPT"),
            timestamp=parse_datetime_now(),
            predicted_patients=float(predicted_patients),
            objective=float(summary.get("objective")) if summary.get("objective") is not None else None,
            summary_json=json.dumps(summary, ensure_ascii=False),
            allocations_json=json.dumps(result.get("department_allocations", []), ensure_ascii=False),
            actions_json=json.dumps(result.get("actions", []), ensure_ascii=False),
            recommendations_json=json.dumps(result.get("recommendations", []), ensure_ascii=False),
        )
        db.add(row)
        db.commit()
    except Exception as e:
        db.rollback()
        logging.exception("Failed to persist optimization run: %s", e)

    return result


@ml_router.get("/optimization_runs")
def list_optimization_runs(
    limit: int = Query(default=20, ge=1, le=200),
    _token: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    rows = db.query(OptimizationRun).order_by(OptimizationRun.id.desc()).limit(limit).all()
    payload = []
    for r in rows:
        payload.append(
            {
                "run_id": normalize_text(r.run_id),
                "timestamp": normalize_text(r.timestamp),
                "predicted_patients": float(r.predicted_patients),
                "objective": float(r.objective) if r.objective is not None else None,
                "summary": json.loads(r.summary_json) if r.summary_json else {},
            }
        )
    return {"runs": payload}


@ml_router.get("/optimization_runs/{run_id}")
def get_optimization_run(
    run_id: str,
    _token: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    row = db.query(OptimizationRun).filter(OptimizationRun.run_id == normalize_text(run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Optimization run not found")
    return {
        "run_id": normalize_text(row.run_id),
        "timestamp": normalize_text(row.timestamp),
        "predicted_patients": float(row.predicted_patients),
        "objective": float(row.objective) if row.objective is not None else None,
        "summary": json.loads(row.summary_json) if row.summary_json else {},
        "department_allocations": json.loads(row.allocations_json) if row.allocations_json else [],
        "actions": json.loads(row.actions_json) if row.actions_json else [],
        "recommendations": json.loads(row.recommendations_json) if row.recommendations_json else [],
    }


@ml_router.post("/predict")
def predict(
    payload: PredictRequest,
    _token: dict = Depends(require_staff_or_admin),
):
    sequence_array = np.array(payload.sequence, dtype=float)

    if not validate_sequence_shape(sequence_array):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid sequence shape. Expected "
                f"({SEQUENCE_LENGTH}, {FEATURE_COUNT}), got {sequence_array.shape}."
            ),
        )

    result = predict_hybrid(sequence_array)
    predicted_patients = float(result["hybrid_prediction"])

    return {
        "predicted_patients_next_hour": predicted_patients,
        "emergency_level": predict_emergency_load(predicted_patients),
        "recommended_resources": calculate_recommended_resources(predicted_patients),
        **result,
    }


@ml_router.post("/simulate")
def simulate(
    payload: SimulateRequest,
    _token: dict = Depends(require_staff_or_admin),
):
    adjusted_patients = float(payload.predicted_patients) * (
        1.0 + float(payload.demand_increase_percent) / 100.0
    )

    recommended_resources = calculate_recommended_resources(adjusted_patients)
    bed_allocation = allocate_beds(int(np.ceil(adjusted_patients)), int(payload.beds_available))
    doctor_shortage = max(
        0,
        int(recommended_resources["doctors_needed"]) - int(payload.doctors_available),
    )

    return {
        "simulated_patients": float(round(adjusted_patients, 2)),
        "demand_increase_percent": float(payload.demand_increase_percent),
        "emergency_level": predict_emergency_load(adjusted_patients),
        "bed_allocation": bed_allocation,
        "recommended_resources": recommended_resources,
        "doctor_shortage": int(doctor_shortage),
    }


@ml_router.post("/explain")
def explain(
    payload: ExplainRequest,
    _token: dict = Depends(require_staff_or_admin),
):
    sequence_array = np.array(payload.sequence, dtype=float)

    if not validate_sequence_shape(sequence_array):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid sequence shape. Expected "
                f"({SEQUENCE_LENGTH}, {FEATURE_COUNT}), got {sequence_array.shape}."
            ),
        )

    return explain_feature_importance(sequence_array)


@ml_router.post("/evaluate")
def evaluate(
    payload: EvaluateRequest,
    _token: dict = Depends(require_admin),
):
    return compare_models(
        actual=payload.actual,
        lstm=payload.lstm,
        arimax=payload.arimax,
        hybrid=payload.hybrid,
    )


@upload_router.post("/patient_flow")
def upload_patient_flow(
    file: UploadFile = File(...),
    _token: dict = Depends(require_admin),
):
    ingest_patient_flow(file.file)
    return {"status": "patient flow uploaded"}


@upload_router.post("/appointments")
def upload_appointments(
    file: UploadFile = File(...),
    _token: dict = Depends(require_admin),
):
    ingest_appointments(file.file)
    return {"status": "appointments uploaded"}


@upload_router.post("/or")
def upload_or(
    file: UploadFile = File(...),
    _token: dict = Depends(require_admin),
):
    ingest_or(file.file)
    return {"status": "or bookings uploaded"}


# Backwards-compatible aliases (keep existing dashboard client paths working)
# - Old: GET /message_templates  -> New: GET /messages/templates
@system_router.get("/message_templates", include_in_schema=False)
def _legacy_message_templates(_token: dict = Depends(require_staff_or_admin)):
    return {
        "admin_templates": ADMIN_MESSAGE_TEMPLATES,
        "staff_quick_replies": STAFF_QUICK_REPLIES,
    }


# - Old: GET /users -> New: GET /auth/users
@system_router.get("/users", include_in_schema=False)
def _legacy_users(_token: dict = Depends(require_admin), db: Session = Depends(get_db)):
    # Mirror /auth/users payload.
    users = db.query(User).all()
    return {
        "users": [
            {
                "username": normalize_text(u.username),
                "name": normalize_text(u.name),
                "role": normalize_text(u.role),
                "department": normalize_text(u.department),
            }
            for u in users
        ]
    }


app.include_router(system_router)
app.include_router(auth_router)
app.include_router(messages_router)
app.include_router(patient_flow_router)
app.include_router(ml_router)
app.include_router(upload_router)