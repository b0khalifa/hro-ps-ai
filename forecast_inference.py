"""Canonical forecasting inference logic for HRO-PS.

This module is the single source of truth for *runtime* inference.
Both the FastAPI service and offline evaluation should call into this module
to guarantee alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from artifacts import artifact_diagnostics, get_artifact_paths
from feature_spec import ARIMAX_EXOG_COLUMNS, FEATURE_COLUMNS


@dataclass(frozen=True)
class ForecastAssets:
    lstm_model: Any
    arimax_model: Any
    x_scaler: Any
    y_scaler: Any
    lstm_weight: float
    arimax_weight: float
    scaler_expected_features: int


def _load_hybrid_weights() -> Tuple[float, float]:
    paths = get_artifact_paths()
    import json

    payload = json.loads(paths.hybrid_config.read_text(encoding="utf-8"))
    return float(payload.get("lstm_weight", 0.90)), float(payload.get("arimax_weight", 0.10))


@lru_cache(maxsize=1)
def load_assets() -> ForecastAssets:
    diag = artifact_diagnostics()
    if diag.get("missing"):
        raise FileNotFoundError(f"Missing required artifacts: {diag['missing']}")

    paths = get_artifact_paths()
    lstm_model = load_model(str(paths.lstm_model), compile=False)
    arimax_model = joblib.load(str(paths.arimax_model))
    x_scaler = joblib.load(str(paths.x_scaler))
    y_scaler = joblib.load(str(paths.y_scaler))
    lstm_w, arimax_w = _load_hybrid_weights()

    expected = int(getattr(x_scaler, "n_features_in_", len(FEATURE_COLUMNS)))
    return ForecastAssets(
        lstm_model=lstm_model,
        arimax_model=arimax_model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        lstm_weight=float(lstm_w),
        arimax_weight=float(arimax_w),
        scaler_expected_features=expected,
    )


def validate_sequence_shape(sequence_array: np.ndarray, sequence_length: int) -> None:
    if sequence_array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {sequence_array.shape}")
    if sequence_array.shape != (sequence_length, len(FEATURE_COLUMNS)):
        raise ValueError(
            f"Invalid sequence shape. Expected ({sequence_length}, {len(FEATURE_COLUMNS)}), got {sequence_array.shape}."
        )


def _scale_sequence(sequence_array: np.ndarray, assets: ForecastAssets) -> np.ndarray:
    flat = sequence_array.reshape(-1, sequence_array.shape[-1])
    if flat.shape[1] != int(assets.scaler_expected_features):
        raise ValueError(
            f"X has {flat.shape[1]} features, but scaler expects {assets.scaler_expected_features}."
        )
    scaled_flat = assets.x_scaler.transform(flat)
    return scaled_flat.reshape(sequence_array.shape).astype(np.float32)


def _inverse_scale_target(pred_scaled: float, assets: ForecastAssets) -> float:
    value = np.array([[float(pred_scaled)]], dtype=np.float32)
    return float(assets.y_scaler.inverse_transform(value)[0][0])


def _get_next_exog_from_sequence(sequence_array: np.ndarray) -> np.ndarray:
    feature_index = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}
    last_row = sequence_array[-1]
    return np.array([[last_row[feature_index[col]] for col in ARIMAX_EXOG_COLUMNS]], dtype=float)


def predict_hybrid(sequence_array: np.ndarray) -> Dict[str, float]:
    assets = load_assets()

    scaled_sequence = _scale_sequence(sequence_array, assets)
    x_input = np.array([scaled_sequence], dtype=np.float32)
    pred_scaled = float(assets.lstm_model.predict(x_input, verbose=0)[0][0])
    lstm_pred = _inverse_scale_target(pred_scaled, assets)

    next_exog = _get_next_exog_from_sequence(sequence_array)
    forecast = assets.arimax_model.forecast(steps=1, exog=next_exog)
    arimax_pred = float(forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0])

    hybrid = assets.lstm_weight * lstm_pred + assets.arimax_weight * arimax_pred
    return {
        "lstm_prediction": float(lstm_pred),
        "arimax_prediction": float(arimax_pred),
        "hybrid_prediction": float(hybrid),
        "lstm_weight": float(assets.lstm_weight),
        "arimax_weight": float(assets.arimax_weight),
    }
