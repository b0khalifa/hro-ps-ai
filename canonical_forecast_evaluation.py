"""Canonical forecast evaluation aligned with runtime inference.

This script intentionally shares the same:
- feature engineering (`forecast_features`)
- sequence format (`feature_spec.FEATURE_COLUMNS` / `SEQUENCE_LENGTH`)
- inference logic (`forecast_inference.predict_hybrid`)

It evaluates 1-step ahead predictions over a horizon window from engineered_data.csv
and reports MAE/RMSE for LSTM, ARIMAX, and Hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from feature_spec import FEATURE_COLUMNS, SEQUENCE_LENGTH
from forecast_inference import predict_hybrid


ENGINEERED_FILE = "engineered_data.csv"


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _load_engineered_df(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run feature_engineering.py first.")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"engineered file missing columns: {missing}")
    for c in FEATURE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def evaluate_one_step(df: pd.DataFrame, start_index: int, end_index: int) -> Dict[str, float]:
    """Evaluate 1-step ahead over indices [start_index, end_index)."""

    y_true: List[float] = []
    y_lstm: List[float] = []
    y_arimax: List[float] = []
    y_hybrid: List[float] = []

    for t in range(start_index, end_index):
        seq_start = t - SEQUENCE_LENGTH
        seq_end = t
        if seq_start < 0:
            continue

        seq = df.loc[seq_start : seq_end - 1, FEATURE_COLUMNS].values.astype(float)
        result = predict_hybrid(seq)

        y_true.append(float(df.loc[t, "patients"]))
        y_lstm.append(float(result["lstm_prediction"]))
        y_arimax.append(float(result["arimax_prediction"]))
        y_hybrid.append(float(result["hybrid_prediction"]))

    y_true_arr = np.array(y_true, dtype=float)
    metrics = {
        "count": float(len(y_true_arr)),
        "lstm_mae": mae(y_true_arr, np.array(y_lstm, dtype=float)),
        "lstm_rmse": rmse(y_true_arr, np.array(y_lstm, dtype=float)),
        "arimax_mae": mae(y_true_arr, np.array(y_arimax, dtype=float)),
        "arimax_rmse": rmse(y_true_arr, np.array(y_arimax, dtype=float)),
        "hybrid_mae": mae(y_true_arr, np.array(y_hybrid, dtype=float)),
        "hybrid_rmse": rmse(y_true_arr, np.array(y_hybrid, dtype=float)),
    }
    return metrics


def main():
    df = _load_engineered_df(ENGINEERED_FILE)
    # Evaluate last 7 days worth of hourly points if available; fallback to last 500.
    window = min(len(df) - SEQUENCE_LENGTH - 1, 24 * 7)
    if window <= 0:
        raise ValueError("Not enough rows to evaluate.")
    start = len(df) - window
    end = len(df)

    metrics = evaluate_one_step(df, start_index=start, end_index=end)
    print("Canonical evaluation (aligned with runtime artifacts)")
    for k in [
        "count",
        "lstm_mae",
        "lstm_rmse",
        "arimax_mae",
        "arimax_rmse",
        "hybrid_mae",
        "hybrid_rmse",
    ]:
        print(f"{k}: {metrics[k]:.4f}" if k != "count" else f"{k}: {int(metrics[k])}")


if __name__ == "__main__":
    main()
