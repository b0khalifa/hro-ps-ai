from typing import List, Optional
import json
import os
from datetime import datetime
import traceback

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model

from resource_optimizer import optimize_resources
from database import get_db
from models import User
from schemas import LoginRequest

app = FastAPI(title="Hospital AI API")

MESSAGES_FILE = "messages_log.csv"
ENGINEERED_FILE = "engineered_data.csv"
SEQUENCE_LENGTH = 24

MESSAGE_COLS = [
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

lstm_model = load_model("hospital_forecast_model.keras", compile=False)
arimax_model = joblib.load("arimax_model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

with open("hybrid_config.json", "r", encoding="utf-8") as f:
    hybrid_config = json.load(f)

HYBRID_LSTM_WEIGHT = float(hybrid_config.get("lstm_weight", 0.9))
HYBRID_ARIMAX_WEIGHT = float(hybrid_config.get("arimax_weight", 0.1))

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


def load_feature_columns() -> List[str]:
    # 1) Preferred: read directly from scaler if available
    if hasattr(x_scaler, "feature_names_in_"):
        cols = list(x_scaler.feature_names_in_)
        if cols:
            return cols

    # 2) Fallback: read from feature engineering metadata
    metadata_file = "feature_engineering_metadata.json"
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            for key in [
                "feature_columns",
                "final_feature_columns",
                "model_feature_columns",
                "training_feature_columns",
            ]:
                cols = meta.get(key)
                if isinstance(cols, list) and len(cols) > 0:
                    return cols
        except Exception:
            pass

    # 3) Fallback: infer from engineered_data.csv by removing non-feature columns
    if os.path.exists(ENGINEERED_FILE):
        try:
            df_tmp = pd.read_csv(ENGINEERED_FILE, nrows=5)
            exclude_cols = {
                "datetime",
                "date",
                "timestamp",
                "target",
                "y",
            }
            cols = [c for c in df_tmp.columns if c not in exclude_cols]
            if len(cols) >= 2:
                return cols
        except Exception:
            pass

    # 4) Last fallback: use a fixed known feature list for your v2 pipeline
    return [
        "patients",
        "day_of_week",
        "month",
        "is_weekend",
        "holiday",
        "weather",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "lag_24",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_mean_6",
        "rolling_std_6",
        "rolling_mean_12",
        "rolling_std_12",
        "rolling_mean_24",
        "rolling_std_24",
        "diff_1",
        "diff_24",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]


FEATURE_COLUMNS = load_feature_columns()
FEATURE_COUNT = len(FEATURE_COLUMNS)
FEATURE_NAMES = FEATURE_COLUMNS.copy()


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


def ensure_messages_file():
    if not os.path.exists(MESSAGES_FILE):
        pd.DataFrame(columns=MESSAGE_COLS).to_csv(MESSAGES_FILE, index=False)


def load_messages_df() -> pd.DataFrame:
    ensure_messages_file()
    df = pd.read_csv(MESSAGES_FILE)
    for col in MESSAGE_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[MESSAGE_COLS].copy()


def save_messages_df(df: pd.DataFrame):
    for col in MESSAGE_COLS:
        if col not in df.columns:
            df[col] = ""
    df[MESSAGE_COLS].to_csv(MESSAGES_FILE, index=False)


def validate_sequence_shape(arr: np.ndarray):
    return arr.shape == (SEQUENCE_LENGTH, FEATURE_COUNT)


def scale_sequence(sequence_array: np.ndarray):
    flat = sequence_array.reshape(-1, sequence_array.shape[-1])
    scaled_flat = x_scaler.transform(flat)
    return scaled_flat.reshape(sequence_array.shape).astype(np.float32)


def inverse_scale_target(pred_scaled: float):
    value = np.array([[pred_scaled]], dtype=np.float32)
    return float(y_scaler.inverse_transform(value)[0][0])


def get_next_exog_from_sequence(sequence_array: np.ndarray):
    last_row = sequence_array[-1]
    exog_cols = FEATURE_COLUMNS[1:]
    exog_values = last_row[1:]
    return pd.DataFrame([exog_values], columns=exog_cols)


def predict_lstm(sequence_array: np.ndarray):
    scaled_sequence = scale_sequence(sequence_array)
    x_input = np.array([scaled_sequence], dtype=np.float32)
    pred_scaled = float(lstm_model.predict(x_input, verbose=0)[0][0])
    return inverse_scale_target(pred_scaled)


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
        "lstm_prediction": lstm_pred,
        "arimax_prediction": arimax_pred,
        "hybrid_prediction": hybrid_prediction,
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


def explain_feature_importance(sequence_array: np.ndarray):
    base_result = predict_hybrid(sequence_array)
    base_pred = float(base_result["hybrid_prediction"])
    impacts = []

    for i, feature_name in enumerate(FEATURE_NAMES):
        modified = sequence_array.copy()
        if feature_name == "patients":
            modified[-1, i] = modified[-1, i] * 1.10
        else:
            modified[-1, i] = modified[-1, i] + 1

        new_result = predict_hybrid(modified)
        new_pred = float(new_result["hybrid_prediction"])
        impacts.append({
            "feature": feature_name,
            "impact": new_pred - base_pred,
        })

    impacts = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)
    return {
        "base_prediction": base_pred,
        "feature_impacts": impacts,
    }


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


@app.get("/debug/predict_info")
def debug_predict_info():
    return {
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "scaler_expected_features": int(getattr(x_scaler, "n_features_in_", -1)),
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
):
    df = load_messages_df()

    if role:
        role = str(role).strip().lower()
        df = df[
            (df["target_role"].astype(str).str.lower() == role)
            | (df["target_role"].astype(str).str.lower() == "all")
        ]

    if department:
        department = str(department).strip().lower()
        df = df[
            (df["target_department"].astype(str).str.lower() == department)
            | (df["target_department"].astype(str).str.lower() == "all departments")
            | (df["target_department"].astype(str).str.lower() == "all")
        ]

    df = df.sort_values(by="timestamp", ascending=False).head(limit)
    return {"messages": df.to_dict(orient="records")}


@app.post("/messages/send")
def send_message(payload: SendMessageRequest):
    df = load_messages_df()

    row = {
        "message_id": f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sender_role": payload.sender_role,
        "sender_name": payload.sender_name,
        "target_role": payload.target_role,
        "target_department": payload.target_department,
        "priority": payload.priority,
        "category": payload.category,
        "title": payload.title,
        "message": payload.message,
        "status": "sent",
        "reply": "",
        "reply_by": "",
        "reply_timestamp": "",
        "acknowledged": "no",
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_messages_df(df)

    return {"status": "sent", "message_id": row["message_id"]}


@app.post("/messages/reply")
def reply_to_message(payload: ReplyMessageRequest):
    df = load_messages_df()
    mask = df["message_id"].astype(str) == str(payload.message_id)

    if not mask.any():
        raise HTTPException(status_code=404, detail="Message not found")

    df.loc[mask, "reply"] = payload.reply
    df.loc[mask, "reply_by"] = payload.reply_by
    df.loc[mask, "reply_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[mask, "acknowledged"] = "yes"
    df.loc[mask, "status"] = "replied"

    save_messages_df(df)
    return {"status": "updated", "message_id": payload.message_id}


@app.get("/optimize_resources/{predicted_patients}")
def optimize_resources_endpoint(predicted_patients: float):
    return optimize_resources(predicted_patients)


@app.post("/predict")
def predict(data: PredictRequest):
    try:
        arr = np.array(data.sequence, dtype=float)

        if not validate_sequence_shape(arr):
            raise HTTPException(
                status_code=400,
                detail=f"Input must be shape ({SEQUENCE_LENGTH}, {FEATURE_COUNT})"
            )

        pred_result = predict_hybrid(arr)
        hybrid_pred = float(pred_result["hybrid_prediction"])
        optimization_result = optimize_resources(hybrid_pred)
        summary = optimization_result["summary"]
        emergency = predict_emergency_load(hybrid_pred)

        return {
            "predicted_patients_next_hour": hybrid_pred,
            "lstm_prediction": pred_result["lstm_prediction"],
            "arimax_prediction": pred_result["arimax_prediction"],
            "hybrid_prediction": hybrid_pred,
            "hybrid_weights": {
                "lstm": pred_result["lstm_weight"],
                "arimax": pred_result["arimax_weight"],
            },
            "emergency_level": emergency,
            "recommended_resources": {
                "beds_needed": summary["beds_needed_total"],
                "doctors_needed": summary["doctors_needed_total"],
                "nurses_needed": summary["nurses_needed_total"],
            },
            "optimization_summary": summary,
            "department_allocations": optimization_result["department_allocations"],
            "optimization_recommendations": optimization_result["recommendations"],
        }

    except HTTPException:
        raise
    except Exception as e:
        print("\n====== /predict INTERNAL ERROR ======")
        traceback.print_exc()
        print("====================================\n")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate")
def simulate(data: SimulateRequest):
    simulated_patients = data.predicted_patients * (1 + data.demand_increase_percent / 100)

    optimization_result = optimize_resources(simulated_patients)
    summary = optimization_result["summary"]
    bed_result = allocate_beds(int(np.ceil(simulated_patients)), data.beds_available)
    emergency = predict_emergency_load(simulated_patients)

    doctor_shortage = max(0, int(summary["doctors_needed_total"]) - data.doctors_available)

    return {
        "simulated_patients": float(simulated_patients),
        "emergency_level": emergency,
        "bed_allocation": bed_result,
        "recommended_resources": {
            "beds_needed": summary["beds_needed_total"],
            "doctors_needed": summary["doctors_needed_total"],
            "nurses_needed": summary["nurses_needed_total"],
        },
        "optimization_summary": summary,
        "department_allocations": optimization_result["department_allocations"],
        "optimization_recommendations": optimization_result["recommendations"],
        "doctor_shortage": doctor_shortage,
    }


@app.post("/explain")
def explain(data: ExplainRequest):
    arr = np.array(data.sequence, dtype=float)

    if not validate_sequence_shape(arr):
        raise HTTPException(
            status_code=400,
            detail=f"Input must be shape ({SEQUENCE_LENGTH}, {FEATURE_COUNT})"
        )

    return explain_feature_importance(arr)


@app.post("/auth/login")
def login_user(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()

    if not user or user.password != payload.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "username": user.username,
        "name": user.name,
        "role": user.role,
        "department": user.department,
    }


@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        {
            "username": u.username,
            "name": u.name,
            "role": u.role,
            "department": u.department,
        }
        for u in users
    ]


@app.get("/patient_flow/latest")
def get_latest_patient_flow():
    if not os.path.exists(ENGINEERED_FILE):
        raise HTTPException(status_code=404, detail=f"{ENGINEERED_FILE} not found")

    df = pd.read_csv(ENGINEERED_FILE)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing feature columns: {missing}")

    df = df[FEATURE_COLUMNS].copy()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    if len(df) < SEQUENCE_LENGTH:
        raise HTTPException(status_code=404, detail="Not enough rows for latest sequence")

    sequence = df.tail(SEQUENCE_LENGTH).values.astype(float).tolist()
    return {"sequence": sequence}