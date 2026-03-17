from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


INPUT_FILE = "engineered_data.csv"
OUTPUT_FILE = "prepared_sequences_v2.npz"
X_SCALER_FILE = "x_scaler.pkl"
Y_SCALER_FILE = "y_scaler.pkl"
SPLIT_INFO_FILE = "split_info.json"
FEATURE_INFO_FILE = "feature_columns.json"

SEQUENCE_LENGTH = 24
TARGET_COL = "patients"

FEATURE_COLUMNS = [
    "patients",
    "day_of_week",
    "month",
    "is_weekend",
    "holiday",
    "weather",
    "hour",
    "hour_sin",
    "hour_cos",
    "patients_lag_1",
    "patients_lag_2",
    "patients_lag_3",
    "patients_lag_6",
    "patients_lag_12",
    "patients_lag_24",
    "patients_roll_mean_3",
    "patients_roll_mean_6",
    "patients_roll_mean_12",
    "patients_roll_mean_24",
    "patients_roll_std_3",
    "patients_roll_std_6",
    "patients_roll_std_12",
    "patients_roll_std_24",
    "patients_diff_1",
    "patients_diff_24",
    "trend_feature",
]


def load_data() -> pd.DataFrame:
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run feature_engineering.py first.")

    df = pd.read_csv(INPUT_FILE)

    missing = [col for col in FEATURE_COLUMNS + [TARGET_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_FILE}: {missing}")

    for col in FEATURE_COLUMNS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL]).reset_index(drop=True)

    if len(df) <= SEQUENCE_LENGTH + 10:
        raise ValueError(
            f"Not enough valid rows after loading engineered data. Found {len(df)} rows only."
        )

    return df


def time_based_split(df: pd.DataFrame):
    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df, train_end, val_end, n


def scale_data(train_df, val_df, test_df):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    x_val = val_df[FEATURE_COLUMNS].values.astype(np.float32)
    x_test = test_df[FEATURE_COLUMNS].values.astype(np.float32)

    y_train = train_df[[TARGET_COL]].values.astype(np.float32)
    y_val = val_df[[TARGET_COL]].values.astype(np.float32)
    y_test = test_df[[TARGET_COL]].values.astype(np.float32)

    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    return (
        x_scaler,
        y_scaler,
        x_scaler.transform(x_train),
        x_scaler.transform(x_val),
        x_scaler.transform(x_test),
        y_scaler.transform(y_train),
        y_scaler.transform(y_val),
        y_scaler.transform(y_test),
    )


def build_sequences(x_data, y_data, sequence_length):
    X_seq = []
    y_seq = []

    for i in range(sequence_length, len(x_data)):
        X_seq.append(x_data[i - sequence_length:i])
        y_seq.append(y_data[i])

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
    )


def main():
    df = load_data()

    train_df, val_df, test_df, train_end, val_end, total_rows = time_based_split(df)

    (
        x_scaler,
        y_scaler,
        x_train_scaled,
        x_val_scaled,
        x_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
    ) = scale_data(train_df, val_df, test_df)

    X_train, y_train = build_sequences(x_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_val, y_val = build_sequences(x_val_scaled, y_val_scaled, SEQUENCE_LENGTH)
    X_test, y_test = build_sequences(x_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("One of the train/val/test sequence sets is empty.")

    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    joblib.dump(x_scaler, X_SCALER_FILE)
    joblib.dump(y_scaler, Y_SCALER_FILE)

    split_info = {
        "input_file": INPUT_FILE,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COL,
        "total_rows_after_engineering": total_rows,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_end_index": train_end,
        "val_end_index": val_end,
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
        "X_test_shape": list(X_test.shape),
    }

    with open(SPLIT_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    with open(FEATURE_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": FEATURE_COLUMNS}, f, indent=2)

    print("✅ Sequences prepared successfully.")
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val:   {X_val.shape}, {y_val.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")
    print(f"Saved: {OUTPUT_FILE}, {X_SCALER_FILE}, {Y_SCALER_FILE}, {SPLIT_INFO_FILE}")


if __name__ == "__main__":
    main()