import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ========================================
# CONFIG
# ========================================
DATA_FILE = "clean_data.csv"
MODEL_FILE = "arimax_model.pkl"

REQUIRED_COLS = [
    "patients",
    "day_of_week",
    "month",
    "is_weekend",
    "holiday",
    "weather"
]

# ========================================
# LOAD DATA
# ========================================
df = pd.read_csv(DATA_FILE)

print("\n========== RAW DATA CHECK ==========")
print("Shape before cleaning:", df.shape)
print("Columns:", list(df.columns))

missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in {DATA_FILE}: {missing_cols}")

# Keep only required columns
df = df[REQUIRED_COLS].copy()

print("\n========== BEFORE TYPE CLEANING ==========")
print(df.head())
print("\nNull counts before cleaning:")
print(df.isna().sum())

# ========================================
# FORCE NUMERIC CONVERSION
# ========================================
for col in REQUIRED_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n========== AFTER NUMERIC CONVERSION ==========")
print(df.head())
print("\nNull counts after numeric conversion:")
print(df.isna().sum())

# Drop invalid rows
df = df.dropna().reset_index(drop=True)

print("\n========== AFTER DROPNA ==========")
print("Shape after cleaning:", df.shape)

if df.empty:
    raise ValueError(
        "clean_data.csv became empty after cleaning. "
        "Check if one or more columns contain non-numeric or missing values."
    )

if len(df) < 30:
    raise ValueError(
        f"Not enough rows to train ARIMAX reliably. Found only {len(df)} rows."
    )

# ========================================
# TARGET + EXOG
# ========================================
y = df["patients"].astype(float)

# IMPORTANT:
# do NOT include patients inside exogenous variables
exog = df[["day_of_week", "month", "is_weekend", "holiday", "weather"]].astype(float)

print("\n========== TRAINING MATRICES ==========")
print("y shape:", y.shape)
print("exog shape:", exog.shape)
print("exog sample:")
print(exog.head())

if exog.empty or exog.shape[1] == 0:
    raise ValueError("Exogenous feature matrix is empty.")

# ========================================
# TRAIN ARIMAX
# ========================================
print("\n========== TRAINING ARIMAX ==========")

model = SARIMAX(
    endog=y,
    exog=exog,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

# ========================================
# SAVE MODEL
# ========================================
joblib.dump(results, MODEL_FILE)

print(f"\nARIMAX model trained successfully and saved as {MODEL_FILE}")