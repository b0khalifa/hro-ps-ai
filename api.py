from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from database import get_db
from models import User, PatientFlow, Appointment, ORBooking, StaffShift, RecommendationLog
from schemas import LoginRequest

app = FastAPI(title="Hospital AI API")

# ========================================
# LOAD MODELS
# ========================================
lstm_model = load_model("hospital_forecast_model.keras", compile=False)
arimax_model = joblib.load("arimax_model.pkl")

FEATURE_NAMES = [
    "patients",
    "day_of_week",
    "month",
    "is_weekend",
    "holiday",
    "weather"
]

HYBRID_LSTM_WEIGHT = 0.6
HYBRID_ARIMAX_WEIGHT = 0.4


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


# ========================================
# HELPERS
# ========================================
def optimize_resources(predicted_patients: float):
    beds_needed = int(predicted_patients * 1.1)
    doctors_needed = max(1, int(predicted_patients / 8))
    nurses_needed = max(1, int(predicted_patients / 4))

    return {
        "beds_needed": beds_needed,
        "doctors_needed": doctors_needed,
        "nurses_needed": nurses_needed
    }


def predict_emergency_load(predicted_patients: float):
    if predicted_patients < 80:
        return "LOW"
    elif predicted_patients < 120:
        return "MEDIUM"
    else:
        return "HIGH"


def allocate_beds(predicted_patients: int, available_beds: int):
    if predicted_patients <= available_beds:
        return {
            "status": "OK",
            "beds_used": predicted_patients,
            "beds_remaining": available_beds - predicted_patients,
            "shortage": 0
        }
    else:
        shortage = predicted_patients - available_beds
        return {
            "status": "SHORTAGE",
            "beds_used": available_beds,
            "beds_remaining": 0,
            "shortage": shortage
        }


def validate_sequence_shape(arr: np.ndarray):
    return arr.shape == (24, 6)


def get_next_exog_from_sequence(sequence_array: np.ndarray):
    last_row = sequence_array[-1]
    exog = np.array([[last_row[1], last_row[2], last_row[3], last_row[4], last_row[5]]], dtype=float)
    return exog


def predict_lstm(sequence_array: np.ndarray):
    X = np.array([sequence_array], dtype=float)
    pred = float(lstm_model.predict(X, verbose=0)[0][0])
    return pred


def predict_arimax(sequence_array: np.ndarray):
    next_exog = get_next_exog_from_sequence(sequence_array)
    forecast = arimax_model.forecast(steps=1, exog=next_exog)
    return float(forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0])


def predict_hybrid(sequence_array: np.ndarray):
    lstm_pred = predict_lstm(sequence_array)
    arimax_pred = predict_arimax(sequence_array)

    LSTM_WEIGHT = 0.65
    ARIMAX_WEIGHT = 0.35

    hybrid_prediction = (LSTM_WEIGHT * lstm_pred) + (ARIMAX_WEIGHT * arimax_pred)

    return {
        "lstm_prediction": lstm_pred,
        "arimax_prediction": arimax_pred,
        "hybrid_prediction": hybrid_pred
    }


def explain_feature_importance(sequence_array: np.ndarray):
    base_result = predict_hybrid(sequence_array)
    base_pred = float(base_result["hybrid_prediction"])

    impacts = []

    for i, feature_name in enumerate(FEATURE_NAMES):
        modified = sequence_array.copy()

        if feature_name == "patients":
            modified[-1, i] = modified[-1, i] * 1.10
        elif feature_name in ["day_of_week", "month", "weather"]:
            modified[-1, i] = modified[-1, i] + 1
        else:
            modified[-1, i] = 1 - modified[-1, i]

        new_result = predict_hybrid(modified)
        new_pred = float(new_result["hybrid_prediction"])
        impact = new_pred - base_pred

        impacts.append({
            "feature": feature_name,
            "impact": impact
        })

    impacts = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "base_prediction": base_pred,
        "feature_impacts": impacts
    }


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
        "status": "running"
    }


@app.post("/predict")
def predict(data: PredictRequest):
    arr = np.array(data.sequence, dtype=float)

    if not validate_sequence_shape(arr):
        return {"error": "Input must be shape (24, 6)"}

    pred_result = predict_hybrid(arr)
    hybrid_pred = float(pred_result["hybrid_prediction"])

    resources = optimize_resources(hybrid_pred)
    emergency = predict_emergency_load(hybrid_pred)

    return {
        "predicted_patients_next_hour": hybrid_pred,
        "lstm_prediction": pred_result["lstm_prediction"],
        "arimax_prediction": pred_result["arimax_prediction"],
        "hybrid_prediction": hybrid_pred,
        "emergency_level": emergency,
        "recommended_resources": resources
    }


@app.post("/simulate")
def simulate(data: SimulateRequest):
    simulated_patients = data.predicted_patients * (1 + data.demand_increase_percent / 100)

    resources = optimize_resources(simulated_patients)
    bed_result = allocate_beds(int(simulated_patients), data.beds_available)
    emergency = predict_emergency_load(simulated_patients)

    doctor_shortage = max(0, resources["doctors_needed"] - data.doctors_available)

    return {
        "simulated_patients": float(simulated_patients),
        "emergency_level": emergency,
        "bed_allocation": bed_result,
        "recommended_resources": resources,
        "doctor_shortage": doctor_shortage
    }


@app.post("/explain")
def explain(data: ExplainRequest):
    arr = np.array(data.sequence, dtype=float)

    if not validate_sequence_shape(arr):
        return {"error": "Input must be shape (24, 6)"}

    result = explain_feature_importance(arr)
    return result

@app.post("/auth/login")
def login_user(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if user.password != payload.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "username": user.username,
        "name": user.name,
        "role": user.role,
        "department": user.department
    }

@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        {
            "username": u.username,
            "name": u.name,
            "role": u.role,
            "department": u.department
        }
        for u in users
    ]