from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


PREPARED_FILE = "prepared_sequences_v2.npz"
Y_SCALER_FILE = "y_scaler.pkl"

MODEL_FILE = "hospital_forecast_model.keras"
VAL_OUTPUTS_FILE = "lstm_val_outputs.npz"
TEST_OUTPUTS_FILE = "lstm_test_outputs.npz"
METRICS_FILE = "lstm_metrics.json"


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


def inverse_transform(y_scaled, y_scaler):
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    return y_scaler.inverse_transform(y_scaled).reshape(-1)


def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


def main():
    if not Path(PREPARED_FILE).exists():
        raise FileNotFoundError(f"{PREPARED_FILE} not found. Run prepare_sequences_v2.py first.")

    data = np.load(PREPARED_FILE)
    y_scaler = joblib.load(Y_SCALER_FILE)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Loaded prepared sequences.")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # load best saved version
    best_model = load_model(MODEL_FILE, compile=False)

    val_pred_scaled = best_model.predict(X_val, verbose=0).reshape(-1)
    test_pred_scaled = best_model.predict(X_test, verbose=0).reshape(-1)

    y_val_true = inverse_transform(y_val, y_scaler)
    y_test_true = inverse_transform(y_test, y_scaler)
    y_val_pred = inverse_transform(val_pred_scaled, y_scaler)
    y_test_pred = inverse_transform(test_pred_scaled, y_scaler)

    val_metrics = evaluate_predictions(y_val_true, y_val_pred)
    test_metrics = evaluate_predictions(y_test_true, y_test_pred)

    np.savez_compressed(
        VAL_OUTPUTS_FILE,
        y_true=y_val_true,
        y_pred=y_val_pred,
    )

    np.savez_compressed(
        TEST_OUTPUTS_FILE,
        y_true=y_test_true,
        y_pred=y_test_pred,
    )

    metrics = {
        "model": "LSTM",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": {
            "loss": [float(x) for x in history.history.get("loss", [])],
            "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("LSTM Validation metrics:", val_metrics)
    print("LSTM Test metrics:", test_metrics)
    print(f"✅ LSTM model saved to {MODEL_FILE}")
    print(f"✅ LSTM metrics saved to {METRICS_FILE}")
    print(f"✅ Validation outputs saved to {VAL_OUTPUTS_FILE}")
    print(f"✅ Test outputs saved to {TEST_OUTPUTS_FILE}")


if __name__ == "__main__":
    main()