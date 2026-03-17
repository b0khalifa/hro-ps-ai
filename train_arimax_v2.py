from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


INPUT_FILE = "engineered_data.csv"
MODEL_FILE = "arimax_model.pkl"
VAL_OUTPUTS_FILE = "arimax_val_outputs.npz"
TEST_OUTPUTS_FILE = "arimax_test_outputs.npz"
METRICS_FILE = "arimax_metrics.json"

TARGET_COL = "patients"

# IMPORTANT:
# Only use safe exogenous variables that do NOT leak actual patient values
EXOG_COLS = [
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


def mape_safe(y_true, y_pred):
    y_true = np.array(y_true, dtype=float).reshape(-1)
    y_pred = np.array(y_pred, dtype=float).reshape(-1)

    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mape_safe(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }


def load_data():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run feature_engineering.py first.")

    df = pd.read_csv(INPUT_FILE)

    missing = [col for col in [TARGET_COL] + EXOG_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_FILE}: {missing}")

    for col in [TARGET_COL] + EXOG_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COL] + EXOG_COLS).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows left after ARIMAX cleaning.")

    return df


def split_data(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def fit_model(y, exog):
    model = SARIMAX(
        endog=y,
        exog=exog,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return results


def main():
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    y_train = train_df[TARGET_COL].astype(float)
    x_train = train_df[EXOG_COLS].astype(float)

    y_val = val_df[TARGET_COL].astype(float)
    x_val = val_df[EXOG_COLS].astype(float)

    y_test = test_df[TARGET_COL].astype(float)
    x_test = test_df[EXOG_COLS].astype(float)

    print("Training ARIMAX on train split...")
    results = fit_model(y_train, x_train)

    print("Forecasting validation split...")
    val_pred = results.forecast(steps=len(x_val), exog=x_val)
    val_pred = np.array(val_pred, dtype=float).reshape(-1)

    print("Re-fitting ARIMAX on train + val...")
    train_val_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    y_train_val = train_val_df[TARGET_COL].astype(float)
    x_train_val = train_val_df[EXOG_COLS].astype(float)

    results_train_val = fit_model(y_train_val, x_train_val)

    print("Forecasting test split...")
    test_pred = results_train_val.forecast(steps=len(x_test), exog=x_test)
    test_pred = np.array(test_pred, dtype=float).reshape(-1)

    val_metrics = evaluate_predictions(y_val.values, val_pred)
    test_metrics = evaluate_predictions(y_test.values, test_pred)

    np.savez_compressed(
        VAL_OUTPUTS_FILE,
        y_true=y_val.values.astype(float),
        y_pred=val_pred.astype(float),
    )

    np.savez_compressed(
        TEST_OUTPUTS_FILE,
        y_true=y_test.values.astype(float),
        y_pred=test_pred.astype(float),
    )

    print("Training final ARIMAX model on full engineered dataset...")
    y_full = df[TARGET_COL].astype(float)
    x_full = df[EXOG_COLS].astype(float)
    final_results = fit_model(y_full, x_full)
    joblib.dump(final_results, MODEL_FILE)

    metrics = {
        "model": "ARIMAX",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "exog_columns": EXOG_COLS,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print(f"✅ ARIMAX model saved to {MODEL_FILE}")
    print(f"✅ Validation outputs saved to {VAL_OUTPUTS_FILE}")
    print(f"✅ Test outputs saved to {TEST_OUTPUTS_FILE}")
    print(f"✅ Metrics saved to {METRICS_FILE}")


if __name__ == "__main__":
    main()