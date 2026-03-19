from typing import List, Optional, Any, Dict, Tuple
import json
import os
from datetime import datetime
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File, APIRouter
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model
from sqlalchemy import text
from functools import lru_cache

from artifacts import artifact_diagnostics, get_artifact_paths, load_manifest
from feature_spec import FEATURE_COLUMNS, ARIMAX_EXOG_COLUMNS, SEQUENCE_LENGTH
from evaluation_service import compare_models
from database import get_db, init_db, engine
from models import User, PatientFlow, MessageLog
from resource_optimizer import optimize_resources
from schemas import LoginRequest
from etl_pipeline import ingest_patient_flow, ingest_appointments, ingest_or
from auth import verify_password

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

LEGACY_MESSAGES_FILE = "messages_log.csv"
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
FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}


def _load_hybrid_weights() -> Tuple[float, float]:
    paths = get_artifact_paths()
    try:
        payload = json.loads(paths.hybrid_config.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load hybrid_config.json from {paths.hybrid_config}") from e
    lstm_w = float(payload.get("lstm_weight", 0.90))
    arimax_w = float(payload.get("arimax_weight", 0.10))
    return lstm_w, arimax_w


@lru_cache(maxsize=1)
def _load_forecast_assets() -> Dict[str, Any]:
    diag = artifact_diagnostics()
    if diag.get("missing"):
        raise FileNotFoundError(
            f"Missing required artifacts: {diag['missing']} (dir={diag.get('artifact_dir')})"
        )

    paths = get_artifact_paths()
    lstm_model = load_model(str(paths.lstm_model), compile=False)
    arimax_model = joblib.load(str(paths.arimax_model))
    x_scaler = joblib.load(str(paths.x_scaler))
    y_scaler = joblib.load(str(paths.y_scaler))
    lstm_w, arimax_w = _load_hybrid_weights()

    scaler_expected_features = int(getattr(x_scaler, "n_features_in_", FEATURE_COUNT))

    return {
        "lstm_model": lstm_model,
        "arimax_model": arimax_model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "lstm_weight": float(lstm_w),
        "arimax_weight": float(arimax_w),
        "scaler_expected_features": scaler_expected_features,
    }


def _get_assets_or_503() -> Dict[str, Any]:
    try:
        return _load_forecast_assets()
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


def validate_sequence_shape(arr: np.ndarray):
    return arr.shape == (SEQUENCE_LENGTH, FEATURE_COUNT)


def scale_sequence(sequence_array: np.ndarray):
    assets = _get_assets_or_503()
    x_scaler = assets["x_scaler"]
    scaler_expected_features = int(assets["scaler_expected_features"])

    flat = sequence_array.reshape(-1, sequence_array.shape[-1])
    if flat.shape[1] != scaler_expected_features:
        raise ValueError(
            f"X has {flat.shape[1]} features, but MinMaxScaler is expecting "
            f"{scaler_expected_features} features as input."
        )
    scaled_flat = x_scaler.transform(flat)
    return scaled_flat.reshape(sequence_array.shape).astype(np.float32)


def inverse_scale_target(pred_scaled: float):
    assets = _get_assets_or_503()
    y_scaler = assets["y_scaler"]
    value = np.array([[pred_scaled]], dtype=np.float32)
    return float(y_scaler.inverse_transform(value)[0][0])


def get_next_exog_from_sequence(sequence_array: np.ndarray):
    last_row = sequence_array[-1]
    return np.array([[last_row[FEATURE_INDEX[col]] for col in ARIMAX_EXOG_COLUMNS]], dtype=float)


def predict_lstm(sequence_array: np.ndarray):
    assets = _get_assets_or_503()
    lstm_model = assets["lstm_model"]
    scaled_sequence = scale_sequence(sequence_array)
    x_input = np.array([scaled_sequence], dtype=np.float32)
    pred_scaled = float(lstm_model.predict(x_input, verbose=0)[0][0])
    return inverse_scale_target(pred_scaled)


def predict_arimax(sequence_array: np.ndarray):
    assets = _get_assets_or_503()
    arimax_model = assets["arimax_model"]
    next_exog = get_next_exog_from_sequence(sequence_array)
    forecast = arimax_model.forecast(steps=1, exog=next_exog)
    return float(forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0])


def predict_hybrid(sequence_array: np.ndarray):
    assets = _get_assets_or_503()
    lstm_pred = predict_lstm(sequence_array)
    arimax_pred = predict_arimax(sequence_array)
    lstm_w = float(assets["lstm_weight"])
    arimax_w = float(assets["arimax_weight"])
    hybrid_prediction = lstm_w * lstm_pred + arimax_w * arimax_pred
    return {
        "lstm_prediction": float(lstm_pred),
        "arimax_prediction": float(arimax_pred),
        "hybrid_prediction": float(hybrid_prediction),
        "lstm_weight": lstm_w,
        "arimax_weight": arimax_w,
    }


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


def build_engineered_sequence_from_patient_flow(rows: List[PatientFlow]) -> List[List[float]]:
    base_df = pd.DataFrame([
        {
            "patients": float(r.patients),
            "day_of_week": float(r.day_of_week or 0),
            "month": float(r.month or 0),
            "is_weekend": float(r.is_weekend or 0),
            "holiday": float(r.holiday or 0),
            "weather": float(r.weather or 0),
        }
        for r in rows
    ])

    base_df = base_df.reset_index(drop=True)
    base_df["hour"] = base_df.index % 24
    base_df["hour_sin"] = np.sin(2 * np.pi * base_df["hour"] / 24.0)
    base_df["hour_cos"] = np.cos(2 * np.pi * base_df["hour"] / 24.0)

    patients = base_df["patients"]
    base_df["patients_lag_1"] = patients.shift(1)
    base_df["patients_lag_2"] = patients.shift(2)
    base_df["patients_lag_3"] = patients.shift(3)
    base_df["patients_lag_6"] = patients.shift(6)
    base_df["patients_lag_12"] = patients.shift(12)
    base_df["patients_lag_24"] = patients.shift(24)

    shifted = patients.shift(1)
    base_df["patients_roll_mean_3"] = shifted.rolling(3, min_periods=1).mean()
    base_df["patients_roll_std_3"] = shifted.rolling(3, min_periods=2).std()
    base_df["patients_roll_mean_6"] = shifted.rolling(6, min_periods=1).mean()
    base_df["patients_roll_std_6"] = shifted.rolling(6, min_periods=2).std()
    base_df["patients_roll_mean_12"] = shifted.rolling(12, min_periods=1).mean()
    base_df["patients_roll_std_12"] = shifted.rolling(12, min_periods=2).std()
    base_df["patients_roll_mean_24"] = shifted.rolling(24, min_periods=1).mean()
    base_df["patients_roll_std_24"] = shifted.rolling(24, min_periods=2).std()

    base_df["patients_diff_1"] = patients.diff(1)
    base_df["patients_diff_24"] = patients.diff(24)
    base_df["trend_feature"] = (
        np.arange(len(base_df), dtype=float) / float(len(base_df) - 1)
        if len(base_df) > 1 else 0.0
    )

    for col in [c for c in base_df.columns if c.startswith("patients_roll_std_")]:
        base_df[col] = base_df[col].fillna(0.0)

    base_df = base_df.bfill().ffill().fillna(0.0)
    sequence_df = base_df[FEATURE_COLUMNS].tail(SEQUENCE_LENGTH).copy()

    if len(sequence_df) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Could not build latest sequence. Need {SEQUENCE_LENGTH} rows, got {len(sequence_df)}."
        )

    return sequence_df.astype(float).values.tolist()


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
    if db.query(MessageLog).count() > 0 or not os.path.exists(LEGACY_MESSAGES_FILE):
        return

    try:
        legacy_df = pd.read_csv(LEGACY_MESSAGES_FILE)
    except Exception:
        return

    if legacy_df.empty:
        return

    for col in LEGACY_MESSAGE_COLS:
        if col not in legacy_df.columns:
            legacy_df[col] = ""

    legacy_df = legacy_df[LEGACY_MESSAGE_COLS].copy()

    rows_added = 0
    for _, row in legacy_df.iterrows():
        message_id = normalize_text(row.get("message_id"))
        if not message_id:
            continue

        if db.query(MessageLog).filter(MessageLog.message_id == message_id).first():
            continue

        db.add(
            MessageLog(
                message_id=message_id,
                timestamp=normalize_text(row.get("timestamp"), parse_datetime_now()),
                sender_role=normalize_text(row.get("sender_role"), "admin"),
                sender_name=normalize_text(row.get("sender_name"), "System"),
                target_role=normalize_text(row.get("target_role"), "all"),
                target_department=normalize_text(row.get("target_department"), "All Departments"),
                priority=normalize_text(row.get("priority"), "normal"),
                category=normalize_text(row.get("category"), "general"),
                title=normalize_text(row.get("title"), "Untitled"),
                message=normalize_text(row.get("message")),
                status=normalize_text(row.get("status"), "sent"),
                reply=normalize_text(row.get("reply")),
                reply_by=normalize_text(row.get("reply_by")),
                reply_timestamp=normalize_text(row.get("reply_timestamp")),
                acknowledged=normalize_text(row.get("acknowledged"), "no"),
                archived=normalize_bool(row.get("archived"), False),
            )
        )
        rows_added += 1

    if rows_added:
        db.commit()


@app.middleware("http")
async def log_requests(request, call_next):
    logging.info("%s %s", request.method, request.url)
    response = await call_next(request)
    return response


@system_router.get("/")
def home():
    return {"message": "Hospital AI API is running"}


@system_router.get("/health")
def health():
    return {"status": "ok"}


@system_router.get("/health/db")
def health_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"db_unhealthy: {e}")


@system_router.get("/status")
def system_status():
    diag = artifact_diagnostics()
    manifest = load_manifest()
    weights = None

    if not diag.get("missing"):
        try:
            lstm_w, arimax_w = _load_hybrid_weights()
            weights = {"lstm": float(lstm_w), "arimax": float(arimax_w)}
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
def get_feature_config():
    return {
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "arimax_exog_columns": ARIMAX_EXOG_COLUMNS,
    }


@system_router.get("/artifacts/manifest")
def get_artifacts_manifest():
    return load_manifest()


@messages_router.get("/templates")
def get_message_templates():
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
def send_message(payload: SendMessageRequest, db: Session = Depends(get_db)):
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
def reply_to_message(payload: ReplyMessageRequest, db: Session = Depends(get_db)):
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

    db.commit()
    db.refresh(row)

    return {
        "status": "updated",
        "message": "Reply saved successfully.",
        "data": serialize_message_row(row),
    }


@messages_router.post("/ack")
def acknowledge_message(payload: MessageActionRequest, db: Session = Depends(get_db)):
    bootstrap_messages_from_csv_if_needed(db)

    row = db.query(MessageLog).filter(
        MessageLog.message_id == normalize_text(payload.message_id)
    ).first()

    if row is None:
        raise HTTPException(status_code=404, detail="Message not found.")

    row.acknowledged = "yes"
    db.commit()
    db.refresh(row)

    return {"status": "acknowledged", "data": serialize_message_row(row)}


@messages_router.post("/archive")
def archive_message(payload: MessageActionRequest, db: Session = Depends(get_db)):
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

    return {
        "username": normalize_text(user.username),
        "name": normalize_text(user.name),
        "role": normalize_text(user.role),
        "department": normalize_text(user.department),
    }


@auth_router.get("/users")
def get_users(db: Session = Depends(get_db)):
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
def get_latest_patient_flow_sequence(db: Session = Depends(get_db)):
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
def optimize_resources_endpoint(predicted_patients: float):
    return optimize_resources(predicted_patients)


@ml_router.post("/predict")
def predict(payload: PredictRequest):
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
def simulate(payload: SimulateRequest):
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
def explain(payload: ExplainRequest):
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
def evaluate(payload: EvaluateRequest):
    return compare_models(
        actual=payload.actual,
        lstm=payload.lstm,
        arimax=payload.arimax,
        hybrid=payload.hybrid,
    )


@upload_router.post("/patient_flow")
def upload_patient_flow(file: UploadFile = File(...)):
    ingest_patient_flow(file.file)
    return {"status": "patient flow uploaded"}


@upload_router.post("/appointments")
def upload_appointments(file: UploadFile = File(...)):
    ingest_appointments(file.file)
    return {"status": "appointments uploaded"}


@upload_router.post("/or")
def upload_or(file: UploadFile = File(...)):
    ingest_or(file.file)
    return {"status": "or bookings uploaded"}


# Backwards-compatible aliases (keep existing dashboard client paths working)
# - Old: GET /message_templates  -> New: GET /messages/templates
@system_router.get("/message_templates", include_in_schema=False)
def _legacy_message_templates():
    return get_message_templates()


# - Old: GET /users -> New: GET /auth/users
@system_router.get("/users", include_in_schema=False)
def _legacy_users(db: Session = Depends(get_db)):
    return get_users(db=db)


app.include_router(system_router)
app.include_router(auth_router)
app.include_router(messages_router)
app.include_router(patient_flow_router)
app.include_router(ml_router)
app.include_router(upload_router)