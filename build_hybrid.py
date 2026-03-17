from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


LSTM_VAL_FILE = "lstm_val_outputs.npz"
LSTM_TEST_FILE = "lstm_test_outputs.npz"
ARIMAX_VAL_FILE = "arimax_val_outputs.npz"
ARIMAX_TEST_FILE = "arimax_test_outputs.npz"

HYBRID_CONFIG_FILE = "hybrid_config.json"
HYBRID_METRICS_FILE = "hybrid_metrics.json"
FORECAST_EVAL_FILE = "forecast_evaluation.csv"
FORECAST_DETAILED_FILE = "forecast_predictions_detailed.csv"


def mape_safe(y_true, y_pred):
    y_true = np.array(y_true, dtype=float).reshape(-1)
    y_pred = np.array(y_pred, dtype=float).reshape(-1)

    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def metrics_dict(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAPE": float(mape_safe(y_true, y_pred)),
    }


def load_npz_pair(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found.")
    data = np.load(path)
    return data["y_true"].reshape(-1), data["y_pred"].reshape(-1)


def align_series(y_true_a, pred_a, y_true_b, pred_b):
    min_len = min(len(y_true_a), len(pred_a), len(y_true_b), len(pred_b))

    y_true_a = y_true_a[-min_len:]
    pred_a = pred_a[-min_len:]
    y_true_b = y_true_b[-min_len:]
    pred_b = pred_b[-min_len:]

    # choose one target series as common reference
    # if slightly different, average is unnecessary; use A
    return y_true_a, pred_a, y_true_b, pred_b


def find_best_weight(y_true, lstm_pred, arimax_pred):
    best_weight = 0.5
    best_rmse = float("inf")
    best_pred = None

    for w in np.arange(0.0, 1.0001, 0.05):
        hybrid_pred = (w * lstm_pred) + ((1.0 - w) * arimax_pred)
        rmse = mean_squared_error(y_true, hybrid_pred) ** 0.5

        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(round(w, 4))
            best_pred = hybrid_pred.copy()

    return best_weight, float(round(1.0 - best_weight, 4)), best_pred


def main():
    y_val_true_lstm, y_val_lstm = load_npz_pair(LSTM_VAL_FILE)
    y_test_true_lstm, y_test_lstm = load_npz_pair(LSTM_TEST_FILE)

    y_val_true_arimax, y_val_arimax = load_npz_pair(ARIMAX_VAL_FILE)
    y_test_true_arimax, y_test_arimax = load_npz_pair(ARIMAX_TEST_FILE)

    y_val_true, y_val_lstm, _, y_val_arimax = align_series(
        y_val_true_lstm, y_val_lstm, y_val_true_arimax, y_val_arimax
    )
    y_test_true, y_test_lstm, _, y_test_arimax = align_series(
        y_test_true_lstm, y_test_lstm, y_test_true_arimax, y_test_arimax
    )

    best_lstm_weight, best_arimax_weight, y_val_hybrid = find_best_weight(
        y_val_true, y_val_lstm, y_val_arimax
    )

    y_test_hybrid = (
        best_lstm_weight * y_test_lstm + best_arimax_weight * y_test_arimax
    )

    lstm_val_metrics = metrics_dict(y_val_true, y_val_lstm)
    arimax_val_metrics = metrics_dict(y_val_true, y_val_arimax)
    hybrid_val_metrics = metrics_dict(y_val_true, y_val_hybrid)

    lstm_test_metrics = metrics_dict(y_test_true, y_test_lstm)
    arimax_test_metrics = metrics_dict(y_test_true, y_test_arimax)
    hybrid_test_metrics = metrics_dict(y_test_true, y_test_hybrid)

    hybrid_config = {
        "lstm_weight": best_lstm_weight,
        "arimax_weight": best_arimax_weight,
        "selection_metric": "validation_rmse",
        "validation_metrics": hybrid_val_metrics,
        "test_metrics": hybrid_test_metrics,
        "alignment_note": "LSTM and ARIMAX outputs aligned using the shortest common tail length.",
    }

    with open(HYBRID_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(hybrid_config, f, indent=2)

    with open(HYBRID_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "validation": {
                    "LSTM": lstm_val_metrics,
                    "ARIMAX": arimax_val_metrics,
                    "Hybrid": hybrid_val_metrics,
                },
                "test": {
                    "LSTM": lstm_test_metrics,
                    "ARIMAX": arimax_test_metrics,
                    "Hybrid": hybrid_test_metrics,
                },
            },
            f,
            indent=2,
        )

    eval_df = pd.DataFrame([
        {"Model": "LSTM", **lstm_test_metrics},
        {"Model": "ARIMAX", **arimax_test_metrics},
        {"Model": "Hybrid", **hybrid_test_metrics},
    ])
    eval_df.to_csv(FORECAST_EVAL_FILE, index=False)

    detailed_df = pd.DataFrame({
        "time_index": np.arange(len(y_test_true)),
        "actual": y_test_true,
        "lstm_pred": y_test_lstm,
        "arimax_pred": y_test_arimax,
        "hybrid_pred": y_test_hybrid,
    })
    detailed_df.to_csv(FORECAST_DETAILED_FILE, index=False)

    print("Hybrid validation metrics:", hybrid_val_metrics)
    print("Hybrid test metrics:", hybrid_test_metrics)
    print(f"Best weights -> LSTM: {best_lstm_weight} | ARIMAX: {best_arimax_weight}")
    print(f"✅ Saved {HYBRID_CONFIG_FILE}")
    print(f"✅ Saved {HYBRID_METRICS_FILE}")
    print(f"✅ Saved {FORECAST_EVAL_FILE}")
    print(f"✅ Saved {FORECAST_DETAILED_FILE}")


if __name__ == "__main__":
    main()