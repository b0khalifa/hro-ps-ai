"""Canonical forecasting feature engineering for HRO-PS.

This module is the *single source of truth* for feature engineering used by:
- training data generation (feature_engineering.py)
- API inference sequence creation (api.py)
- dashboard fallback (if enabled)
- multi-step roll-forward forecasting (forecast_runtime.py)

Design goals:
- deterministic feature ordering (feature_spec.FEATURE_COLUMNS)
- avoid leakage (no using future patients)
- minimal dependencies (numpy/pandas)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from feature_spec import FEATURE_COLUMNS, SEQUENCE_LENGTH


BASE_COLS = ["patients", "day_of_week", "month", "is_weekend", "holiday", "weather"]


@dataclass(frozen=True)
class EngineeredFrameResult:
    df: pd.DataFrame
    feature_columns: list[str]


def _safe_std(values: Sequence[float]) -> float:
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def coerce_base_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce required base columns to numeric and drop invalid rows."""

    out = df.copy()
    for col in BASE_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["patients"]).reset_index(drop=True)
    for col in [c for c in BASE_COLS if c != "patients"]:
        out[col] = out[col].fillna(0.0)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/hour_sin/hour_cos.

    If a real datetime column exists it will be used; otherwise we synthesize
    hour from row index to keep the pipeline runnable.
    """

    out = df.copy().reset_index(drop=True)

    if "datetime" in out.columns:
        parsed = pd.to_datetime(out["datetime"], errors="coerce")
        if parsed.notna().sum() > 0:
            out["hour"] = parsed.dt.hour.fillna(0).astype(int)
        else:
            out["hour"] = out.index % 24
    else:
        out["hour"] = out.index % 24

    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").fillna(0).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    return out


def add_lags_rolls_diffs_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    patients = out["patients"].astype(float)

    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"patients_lag_{lag}"] = patients.shift(lag)

    shifted = patients.shift(1)
    for window in [3, 6, 12, 24]:
        out[f"patients_roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
        out[f"patients_roll_std_{window}"] = shifted.rolling(window, min_periods=2).std()

    out["patients_diff_1"] = patients.diff(1)
    out["patients_diff_24"] = patients.diff(24)

    out["trend_feature"] = (
        np.arange(len(out), dtype=float) / float(len(out) - 1)
        if len(out) > 1 else 0.0
    )

    std_cols = [c for c in out.columns if c.startswith("patients_roll_std_")]
    for c in std_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out = out.bfill().ffill().fillna(0.0)
    return out


def build_engineered_frame(base_df: pd.DataFrame) -> EngineeredFrameResult:
    """Build engineered dataframe with columns matching FEATURE_COLUMNS."""

    df = coerce_base_numeric(base_df)
    df = add_time_features(df)
    df = add_lags_rolls_diffs_trend(df)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Engineered frame missing required columns: {missing}")

    return EngineeredFrameResult(df=df, feature_columns=list(FEATURE_COLUMNS))


def build_latest_sequence_from_rows(rows: Iterable[dict]) -> list[list[float]]:
    """Build latest SEQUENCE_LENGTH engineered sequence from raw patient_flow rows.

    rows must contain at least the BASE_COLS keys.
    """

    df = pd.DataFrame(list(rows))
    engineered = build_engineered_frame(df).df
    seq_df = engineered[list(FEATURE_COLUMNS)].tail(SEQUENCE_LENGTH).copy()
    if len(seq_df) != SEQUENCE_LENGTH:
        raise ValueError(f"Need {SEQUENCE_LENGTH} rows to build latest sequence, got {len(seq_df)}")
    return seq_df.astype(float).values.tolist()


def roll_sequence_forward(sequence: np.ndarray, predicted_patients: float) -> np.ndarray:
    """Roll an engineered sequence forward by one step.

    This is the canonical roll-forward implementation; it must stay consistent
    with FEATURE_COLUMNS ordering.
    """

    seq = np.array(sequence, dtype=float)
    if seq.ndim != 2 or seq.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected sequence shape (n, {len(FEATURE_COLUMNS)}), got {seq.shape}."
        )

    patients_idx = 0
    day_of_week_idx = FEATURE_COLUMNS.index("day_of_week")
    month_idx = FEATURE_COLUMNS.index("month")
    is_weekend_idx = FEATURE_COLUMNS.index("is_weekend")
    holiday_idx = FEATURE_COLUMNS.index("holiday")
    weather_idx = FEATURE_COLUMNS.index("weather")
    hour_idx = FEATURE_COLUMNS.index("hour")
    hour_sin_idx = FEATURE_COLUMNS.index("hour_sin")
    hour_cos_idx = FEATURE_COLUMNS.index("hour_cos")
    trend_idx = FEATURE_COLUMNS.index("trend_feature")

    last_row = seq[-1].copy()
    prev_row = seq[-2].copy() if len(seq) >= 2 else last_row.copy()

    new_row = last_row.copy()
    new_row[patients_idx] = float(predicted_patients)

    prev_hour = int(round(last_row[hour_idx]))
    next_hour = (prev_hour + 1) % 24
    crossed_day = 1 if next_hour == 0 else 0

    current_dow = int(round(last_row[day_of_week_idx]))
    new_dow = (current_dow + crossed_day) % 7

    new_row[day_of_week_idx] = float(new_dow)
    new_row[month_idx] = float(last_row[month_idx])
    new_row[is_weekend_idx] = 1.0 if new_dow >= 5 else 0.0
    new_row[holiday_idx] = float(last_row[holiday_idx])
    new_row[weather_idx] = float(last_row[weather_idx])

    new_row[hour_idx] = float(next_hour)
    new_row[hour_sin_idx] = float(np.sin(2 * np.pi * next_hour / 24.0))
    new_row[hour_cos_idx] = float(np.cos(2 * np.pi * next_hour / 24.0))

    history = seq[:, patients_idx].tolist() + [float(predicted_patients)]

    def lag(lag_hours: int) -> float:
        idx = len(history) - 1 - lag_hours
        if idx < 0:
            return float(history[0])
        return float(history[idx])

    # Lags
    for lag_hours in [1, 2, 3, 6, 12, 24]:
        col = f"patients_lag_{lag_hours}"
        new_row[FEATURE_COLUMNS.index(col)] = lag(lag_hours)

    # Rolling stats are computed on prior history excluding the just-added point.
    prior = history[:-1] if len(history) > 1 else history

    def window(w: int):
        return prior[-w:] if len(prior) >= w else prior

    for w in [3, 6, 12, 24]:
        mean_col = f"patients_roll_mean_{w}"
        std_col = f"patients_roll_std_{w}"
        new_row[FEATURE_COLUMNS.index(mean_col)] = float(np.mean(window(w)))
        new_row[FEATURE_COLUMNS.index(std_col)] = _safe_std(window(w))

    # Diffs
    new_row[FEATURE_COLUMNS.index("patients_diff_1")] = float(predicted_patients - last_row[patients_idx])
    new_row[FEATURE_COLUMNS.index("patients_diff_24")] = float(predicted_patients - lag(24))

    # Trend: continue with the previous increment.
    trend_step = float(last_row[trend_idx] - prev_row[trend_idx]) if len(seq) >= 2 else 0.001
    if abs(trend_step) < 1e-9:
        trend_step = 0.001
    new_row[trend_idx] = float(last_row[trend_idx] + trend_step)

    return np.vstack([seq[1:], new_row])
