from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="Hospital AI API")

# تحميل الموديل مرة واحدة
model = load_model("hospital_forecast_model.keras", compile=False)


# -------- Request Models --------
class PredictRequest(BaseModel):
    sequence: List[List[float]]
    # لازم تكون 24 صف × 6 features
    # [patients, day_of_week, month, is_weekend, holiday, weather]


class SimulateRequest(BaseModel):
    predicted_patients: float
    beds_available: int
    doctors_available: int
    demand_increase_percent: float = 0


# -------- Utility Functions --------
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


# -------- Routes --------
@app.get("/")
def home():
    return {
        "message": "Hospital AI API is running"
    }


@app.post("/predict")
def predict(data: PredictRequest):
    arr = np.array(data.sequence, dtype=float)

    if arr.shape != (24, 6):
        return {
            "error": "Input must be shape (24, 6)"
        }

    X = np.array([arr])
    pred = model.predict(X, verbose=0)[0][0]

    resources = optimize_resources(pred)
    emergency = predict_emergency_load(pred)

    return {
        "predicted_patients_next_hour": float(pred),
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
@app.get("/status")
def system_status():
    return {
        "system": "Hospital AI",
        "model": "LSTM Forecast",
        "status": "running"
    }