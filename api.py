from typing import List, Optional
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model

from database import get_db, Base, engine
from models import User, PatientFlow, MessageLog
from resource_optimizer import optimize_resources
from schemas import LoginRequest

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Hospital AI API")

# ========================================
# LEGACY FILES (bootstrapping only)
# ========================================
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
]

SEQUENCE_LENGTH = 24

# ========================================
# LOAD MODELS + ARTIFACTS
# ========================================
lstm_model = load_model("hospital_forecast_model.keras", compile=False)
arimax_model = joblib.load("arimax_model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

with open("hybrid_config.json", "r", encoding="utf-8") as f:
    hybrid_config = json.load(f)

HYBRID_LSTM_WEIGHT = float(hybrid_config.get("lstm_weight", 0.90))
HYBRID_ARIMAX_WEIGHT = float(hybrid_config.get("arimax_weight", 0.10))

# ========================================
# FIXED FEATURE SET
# MUST MATCH prepare_sequences_v2.py EXACTLY
# ========================================
FEATURE_COLUMNS = [
    "patients",
    "day_of_week",
    "month",
    "is_weekend",
    "holiday",
    "weather",
    "hour",
    "hour_sin",
    "hour_cos",
    "patients_lag_1",
    "patients_lag_2",
    "patients_lag_3",
    "patients_lag_6",
    "patients_lag_12",
    "patients_lag_24",
    "patients_roll_mean_3",
    "patients_roll_std_3",
    "patients_roll_mean_6",
    "patients_roll_std_6",
    "patients_roll_mean_12",
    "patients_roll_std_12",
    "patients_roll_mean_24",
    "patients_roll_std_24",
    "patients_diff_1",
    "patients_diff_24",
    "trend_feature",
]

# MUST MATCH train_arimax_v2.py EXACTLY
ARIMAX_EXOG_COLUMNS = [
    "day_of_week",
    "month",
    "is_weekend",
    "holiday",
    "weather",
    "hour",
    "hour_sin",
    "hour_cos",
    "trend_feature",
]

FEATURE_COUNT = len(FEATURE_COLUMNS)
FEATURE_NAMES = FEATURE_COLUMNS.copy()
SCALER_EXPECTED_FEATURES = int(getattr(x_scaler, "n_features_in_", FEATURE_COUNT))

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

# ========================================
# REQUEST MODELS
# ========================================
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


# ========================================
# HELPERS
# ========================================
def normalize_text(value, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def parse_datetime_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_sequence_shape(arr: np.ndarray):
    return arr.shape == (SEQUENCE_LENGTH, FEATURE_COUNT)


def scale_sequence(sequence_array: np.ndarray):
    flat = sequence_array.reshape(-1, sequence_array.shape[-1])

    if flat.shape[1] != SCALER_EXPECTED_FEATURES:
        raise ValueError(
            f"X has {flat.shape[1]} features, but MinMaxScaler is expecting "
            f"{SCALER_EXPECTED_FEATURES} features as input."
        )

    scaled_flat = x_scaler.transform(flat)
    return scaled_flat.reshape(sequence_array.shape).astype(np.float32)


def inverse_scale_target(pred_scaled: float):
    value = np.array([[pred_scaled]], dtype=np.float32)
    return float(y_scaler.inverse_transform(value)[0][0])


def get_feature_index_map():
    return {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}


FEATURE_INDEX = get_feature_index_map()


def get_next_exog_from_sequence(sequence_array: np.ndarray):
    last_row = sequence_array[-1]
    exog = np.array(
        [[last_row[FEATURE_INDEX[col]] for col in ARIMAX_EXOG_COLUMNS]],
        dtype=float,
    )
    return exog


def predict_lstm(sequence_array: np.ndarray):
    scaled_sequence = scale_sequence(sequence_array)
    x_input = np.array([scaled_sequence], dtype=np.float32)

    pred_scaled = float(lstm_model.predict(x_input, verbose=0)[0][0])
    pred_original = inverse_scale_target(pred_scaled)
    return pred_original


def predict_arimax(sequence_array: np.ndarray):
    next_exog = get_next_exog_from_sequence(sequence_array)
    forecast = arimax_model.forecast(steps=1, exog=next_exog)
    return float(forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0])


def predict_hybrid(sequence_array: np.ndarray):
    lstm_pred = predict_lstm(sequence_array)
    arimax_pred = predict_arimax(sequence_array)

    hybrid_prediction = (
        HYBRID_LSTM_WEIGHT * lstm_pred
        + HYBRID_ARIMAX_WEIGHT * arimax_pred
    )

    return {
        "lstm_prediction": float(lstm_pred),
        "arimax_prediction": float(arimax_pred),
        "hybrid_prediction": float(hybrid_prediction),
        "lstm_weight": HYBRID_LSTM_WEIGHT,
        "arimax_weight": HYBRID_ARIMAX_WEIGHT,
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
    predicted_patients = float(predicted_patients)
    beds_needed = int(np.ceil(predicted_patients * 1.15))
    doctors_needed = max(1, int(np.ceil(predicted_patients / 6.0)))
    nurses_needed = max(1, int(np.ceil(predicted_patients / 3.5)))

    return {
        "beds_needed": beds_needed,
        "doctors_needed": doctors_needed,
        "nurses_needed": nurses_needed,
    }


def explain_feature_importance(sequence_array: np.ndarray):
    base_result = predict_hybrid(sequence_array)
    base_pred = float(base_result["hybrid_prediction"])

    impacts = []

    for i, feature_name in enumerate(FEATURE_NAMES):
        modified = sequence_array.copy()

        if feature_name == "patients":
            modified[-1, i] = modified[-1, i] * 1.10
        elif feature_name in ["day_of_week", "month", "weather", "hour", "trend_feature"]:
            modified[-1, i] = modified[-1, i] + 1
        elif feature_name in ["is_weekend", "holiday"]:
            modified[-1, i] = 1 - modified[-1, i]
        else:
            modified[-1, i] = modified[-1, i] * 1.05

        new_result = predict_hybrid(modified)
        new_pred = float(new_result["hybrid_prediction"])
        impact = new_pred - base_pred

        impacts.append({
            "feature": feature_name,
            "impact": float(impact),
        })

    impacts = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "base_prediction": float(base_pred),
        "feature_impacts": impacts,
    }


def build_engineered_sequence_from_patient_flow(rows: List[PatientFlow]) -> List[List[float]]:
    base_df = pd.DataFrame([
        {
            "patients": float(r.patients),
            "day_of_week": float(r.day_of_week),
            "month": float(r.month),
            "is_weekend": float(r.is_weekend),
            "holiday": float(r.holiday),
            "weather": float(r.weather),
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

    if len(base_df) > 1:
        base_df["trend_feature"] = np.arange(len(base_df), dtype=float) / float(len(base_df) - 1)
    else:
        base_df["trend_feature"] = 0.0

    std_cols = [c for c in base_df.columns if c.startswith("patients_roll_std_")]
    for col in std_cols:
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
    }


def bootstrap_messages_from_csv_if_needed(db: Session):
    existing_count = db.query(MessageLog).count()
    if existing_count > 0:
        return

    if not os.path.exists(LEGACY_MESSAGES_FILE):
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

        already_exists = (
            db.query(MessageLog)
            .filter(MessageLog.message_id == message_id)
            .first()
        )
        if already_exists:
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
            )
        )
        rows_added += 1

    if rows_added > 0:
        db.commit()


# ========================================
# ROUTES
# ========================================
@app.get("/")
def home():
    return {"message": "Hospital AI API is running"}


@app.get("/status")
def system_status():
    return {
        "system": "Hospital AI",
        "model": "Hybrid Forecast (LSTM + ARIMAX)",
        "status": "running",
        "hybrid_weights": {
            "lstm": HYBRID_LSTM_WEIGHT,
            "arimax": HYBRID_ARIMAX_WEIGHT,
        },
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
    }


@app.get("/feature_config")
def get_feature_config():
    return {
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "arimax_exog_columns": ARIMAX_EXOG_COLUMNS,
    }


@app.get("/debug/predict_info")
def debug_predict_info():
    return {
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "arimax_exog_columns": ARIMAX_EXOG_COLUMNS,
        "scaler_expected_features": SCALER_EXPECTED_FEATURES,
    }


@app.get("/message_templates")
def get_message_templates():
    return {
        "admin_templates": ADMIN_MESSAGE_TEMPLATES,
        "staff_quick_replies": STAFF_QUICK_REPLIES,
    }


@app.get("/messages")
def get_messages(
    role: Optional[str] = Query(default=None),
    department: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    unread_only: bool = Query(default=False),
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

    if unread_only:
        query = query.filter(MessageLog.acknowledged.ilike("no"))

    rows = (
        query.order_by(MessageLog.id.desc())
        .limit(limit)
        .all()
    )

    return {
        "messages": [serialize_message_row(row) for row in rows],
        "quick_replies": STAFF_QUICK_REPLIES,
    }


@app.post("/messages/send")
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
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "status": "sent",
        "message": "Message sent successfully.",
        "data": serialize_message_row(row),
    }


@app.post("/messages/reply")
def reply_to_message(payload: ReplyMessageRequest, db: Session = Depends(get_db)):
    bootstrap_messages_from_csv_if_needed(db)

    row = (
        db.query(MessageLog)
        .filter(MessageLog.message_id == normalize_text(payload.message_id))
        .first()
    )

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


@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()

    return {
        "users": [
            {
                "username": normalize_text(user.username),
                "name": normalize_text(user.name),
                "role": normalize_text(user.role),
                "department": normalize_text(user.department),
            }
            for user in users
        ]
    }


@app.post("/auth/login")
def login_user(payload: LoginRequest, db: Session = Depends(get_db)):
    username = normalize_text(payload.username)
    password = normalize_text(payload.password)

    user = db.query(User).filter(User.username == username).first()

    if user is None or normalize_text(user.password) != password:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    return {
        "username": normalize_text(user.username),
        "name": normalize_text(user.name),
        "role": normalize_text(user.role),
        "department": normalize_text(user.department),
    }


@app.get("/patient_flow/latest")
def get_latest_patient_flow_sequence(db: Session = Depends(get_db)):
    rows = (
        db.query(PatientFlow)
        .order_by(PatientFlow.id.desc())
        .limit(SEQUENCE_LENGTH)
        .all()
    )

    if len(rows) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {SEQUENCE_LENGTH} patient flow rows in DB.",
        )

    rows = list(reversed(rows))
    sequence = build_engineered_sequence_from_patient_flow(rows)

    return {
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": FEATURE_COUNT,
        "sequence": sequence,
    }


@app.get("/optimize_resources/{predicted_patients}")
def optimize_resources_endpoint(predicted_patients: float):
    result = optimize_resources(predicted_patients)
    return result


@app.post("/predict")
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
        "lstm_prediction": result["lstm_prediction"],
        "arimax_prediction": result["arimax_prediction"],
        "hybrid_prediction": result["hybrid_prediction"],
        "lstm_weight": result["lstm_weight"],
        "arimax_weight": result["arimax_weight"],
    }


@app.post("/simulate")
def simulate(payload: SimulateRequest):
    adjusted_patients = float(payload.predicted_patients) * (
        1.0 + (float(payload.demand_increase_percent) / 100.0)
    )

    recommended_resources = calculate_recommended_resources(adjusted_patients)
    bed_allocation = allocate_beds(
        predicted_patients=int(np.ceil(adjusted_patients)),
        available_beds=int(payload.beds_available),
    )

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


@app.post("/explain")
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