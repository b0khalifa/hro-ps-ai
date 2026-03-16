import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# CONFIG
# =========================
DATA_FILE = "clean_data.csv"
LSTM_MODEL_FILE = "hospital_forecast_model.keras"
ARIMAX_MODEL_FILE = "arimax_model.pkl"
OUTPUT_FILE = "forecast_evaluation.csv"

SEQUENCE_LENGTH = 24
FEATURE_COLUMNS = [
    "patients",
    "day_of_week",
    "month",
    "is_weekend",
    "holiday",
    "weather"
]
TARGET_COLUMN = "patients"

HYBRID_LSTM_WEIGHT = 0.6
HYBRID_ARIMAX_WEIGHT = 0.4


# =========================
# METRICS
# =========================
def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_FILE)

missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df[FEATURE_COLUMNS].copy()

for col in FEATURE_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

if len(df) <= SEQUENCE_LENGTH + 10:
    raise ValueError("Not enough rows for evaluation.")

# =========================
# LOAD MODELS
# =========================
lstm_model = load_model(LSTM_MODEL_FILE, compile=False)
arimax_model = joblib.load(ARIMAX_MODEL_FILE)

# =========================
# CREATE TEST WINDOW
# =========================
# آخر 20% للاختبار
test_start_idx = int(len(df) * 0.8)

# لازم نبدأ قبلها بـ 24 صف عشان sequence
start_idx = max(SEQUENCE_LENGTH, test_start_idx)

actuals = []
lstm_preds = []
arimax_preds = []
hybrid_preds = []
time_indices = []

data_values = df[FEATURE_COLUMNS].values.astype(float)

for i in range(start_idx, len(df)):
    sequence = data_values[i - SEQUENCE_LENGTH:i]
    actual = float(data_values[i][0])

    # LSTM
    X = np.array([sequence], dtype=float)
    lstm_pred = float(lstm_model.predict(X, verbose=0)[0][0])

    # ARIMAX
    exog = np.array([[sequence[-1][1], sequence[-1][2], sequence[-1][3], sequence[-1][4], sequence[-1][5]]], dtype=float)
    arimax_forecast = arimax_model.forecast(steps=1, exog=exog)
    arimax_pred = float(arimax_forecast.iloc[0] if hasattr(arimax_forecast, "iloc") else arimax_forecast[0])

    # Hybrid
    hybrid_pred = (HYBRID_LSTM_WEIGHT * lstm_pred) + (HYBRID_ARIMAX_WEIGHT * arimax_pred)

    actuals.append(actual)
    lstm_preds.append(lstm_pred)
    arimax_preds.append(arimax_pred)
    hybrid_preds.append(hybrid_pred)
    time_indices.append(i)

# =========================
# METRICS TABLE
# =========================
results = []

for model_name, preds in [
    ("LSTM", lstm_preds),
    ("ARIMAX", arimax_preds),
    ("Hybrid", hybrid_preds),
]:
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = calculate_mape(actuals, preds)

    results.append({
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

# =========================
# SAVE DETAILED PREDICTIONS
# =========================
detailed_df = pd.DataFrame({
    "time_index": time_indices,
    "actual": actuals,
    "lstm_pred": lstm_preds,
    "arimax_pred": arimax_preds,
    "hybrid_pred": hybrid_preds
})

detailed_df.to_csv("forecast_predictions_detailed.csv", index=False)

print("Evaluation completed successfully.")
print(results_df)