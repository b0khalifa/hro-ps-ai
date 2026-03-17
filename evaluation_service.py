import json
import os
import numpy as np
import pandas as pd


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metric_block(payload: dict, split: str):
    """
    Supports both:
    - {"val_metrics": {...}, "test_metrics": {...}}
    - {"validation": {...}, "test": {...}}
    """
    if payload is None:
        return None

    split = split.lower()

    if split == "validation":
        return payload.get("val_metrics") or payload.get("validation")
    return payload.get("test_metrics") or payload.get("test")


def build_metrics_dataframe(split: str = "test") -> pd.DataFrame:
    split = split.lower()

    lstm_payload = _load_json("lstm_metrics.json")
    arimax_payload = _load_json("arimax_metrics.json")
    hybrid_payload = _load_json("hybrid_metrics.json")

    rows = []

    lstm_block = _extract_metric_block(lstm_payload, split)
    if lstm_block:
        rows.append({
            "Model": "LSTM",
            "MAE": float(lstm_block.get("mae", lstm_block.get("MAE", 0))),
            "RMSE": float(lstm_block.get("rmse", lstm_block.get("RMSE", 0))),
            "MAPE": float(lstm_block.get("mape", lstm_block.get("MAPE", 0))),
        })

    arimax_block = _extract_metric_block(arimax_payload, split)
    if arimax_block:
        rows.append({
            "Model": "ARIMAX",
            "MAE": float(arimax_block.get("mae", arimax_block.get("MAE", 0))),
            "RMSE": float(arimax_block.get("rmse", arimax_block.get("RMSE", 0))),
            "MAPE": float(arimax_block.get("mape", arimax_block.get("MAPE", 0))),
        })

    if hybrid_payload:
        if "validation" in hybrid_payload or "test" in hybrid_payload:
            block = hybrid_payload.get("validation") if split == "validation" else hybrid_payload.get("test")
            if isinstance(block, dict) and "Hybrid" in block:
                block = block["Hybrid"]
        else:
            block = _extract_metric_block(hybrid_payload, split)

        if block:
            rows.append({
                "Model": "Hybrid",
                "MAE": float(block.get("mae", block.get("MAE", 0))),
                "RMSE": float(block.get("rmse", block.get("RMSE", 0))),
                "MAPE": float(block.get("mape", block.get("MAPE", 0))),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def build_detailed_predictions_dataframe(split: str = "test") -> pd.DataFrame:
    split = split.lower()

    lstm_file = "lstm_val_outputs.npz" if split == "validation" else "lstm_test_outputs.npz"
    arimax_file = "arimax_val_outputs.npz" if split == "validation" else "arimax_test_outputs.npz"

    if not os.path.exists(lstm_file) or not os.path.exists(arimax_file):
        return pd.DataFrame()

    lstm_data = np.load(lstm_file)
    arimax_data = np.load(arimax_file)

    lstm_true = np.array(lstm_data["y_true"], dtype=float).reshape(-1)
    lstm_pred = np.array(lstm_data["y_pred"], dtype=float).reshape(-1)

    arimax_true = np.array(arimax_data["y_true"], dtype=float).reshape(-1)
    arimax_pred = np.array(arimax_data["y_pred"], dtype=float).reshape(-1)

    min_len = min(len(lstm_true), len(lstm_pred), len(arimax_true), len(arimax_pred))
    if min_len == 0:
        return pd.DataFrame()

    y_true = lstm_true[-min_len:]
    lstm_pred = lstm_pred[-min_len:]
    arimax_pred = arimax_pred[-min_len:]

    hybrid_cfg = _load_json("hybrid_config.json") or {}
    lstm_weight = float(hybrid_cfg.get("lstm_weight", 0.9))
    arimax_weight = float(hybrid_cfg.get("arimax_weight", 0.1))
    hybrid_pred = (lstm_weight * lstm_pred) + (arimax_weight * arimax_pred)

    return pd.DataFrame({
        "time_index": list(range(min_len)),
        "actual": y_true,
        "lstm_pred": lstm_pred,
        "arimax_pred": arimax_pred,
        "hybrid_pred": hybrid_pred,
    })