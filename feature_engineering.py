from pathlib import Path
import json

import numpy as np
import pandas as pd


INPUT_FILE = "clean_data.csv"
OUTPUT_FILE = "engineered_data.csv"
METADATA_FILE = "feature_engineering_metadata.json"

BASE_REQUIRED_COLS = [
    "patients",
    "day_of_week",
    "month",
    "is_weekend",
    "holiday",
    "weather",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [col for col in BASE_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df.copy()


def force_base_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in BASE_REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=BASE_REQUIRED_COLS).reset_index(drop=True)
    after = len(df)

    print(f"Base cleaning -> before: {before}, after: {after}, dropped: {before - after}")

    if df.empty:
        raise ValueError("All rows became invalid after base numeric cleaning.")

    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = 0
    df["day"] = 1
    df["hour_sin"] = 0.0
    df["hour_cos"] = 1.0

    if "datetime" in df.columns:
        parsed = pd.to_datetime(df["datetime"], errors="coerce")

        if parsed.notna().sum() > 0:
            df["hour"] = parsed.dt.hour.fillna(0).astype(int)
            df["day"] = parsed.dt.day.fillna(1).astype(int)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    return df


def add_trend_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time_index"] = np.arange(len(df), dtype=int)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"patients_lag_{lag}"] = df["patients"].shift(lag)

    for window in [3, 6, 12, 24]:
        shifted = df["patients"].shift(1)
        df[f"patients_roll_mean_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
        df[f"patients_roll_std_{window}"] = shifted.rolling(window=window, min_periods=2).std()

    df["patients_diff_1"] = df["patients"].diff(1)
    df["patients_diff_24"] = df["patients"].diff(24)

    if len(df) > 1:
        df["trend_feature"] = df["time_index"] / float(len(df) - 1)
    else:
        df["trend_feature"] = 0.0

    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    feature_cols_to_clean = [
        "patients",
        "day_of_week",
        "month",
        "is_weekend",
        "holiday",
        "weather",
        "hour",
        "day",
        "hour_sin",
        "hour_cos",
        "time_index",
        "trend_feature",
        "patients_diff_1",
        "patients_diff_24",
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
    ]

    existing_cols = [c for c in feature_cols_to_clean if c in df.columns]

    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill standard deviation NaNs with 0
    std_cols = [c for c in df.columns if c.startswith("patients_roll_std_")]
    for col in std_cols:
        df[col] = df[col].fillna(0.0)

    # instead of dropping all rows with any NaN across the whole dataframe,
    # keep only rows where required training features are valid
    before = len(df)
    df = df.dropna(subset=existing_cols).reset_index(drop=True)
    after = len(df)

    print(f"Finalize -> before: {before}, after: {after}, dropped: {before - after}")

    if df.empty:
        # helpful diagnostics
        null_counts = df.isna().sum().to_dict() if len(df) else {}
        raise ValueError(
            "engineered_data.csv would be empty after feature engineering and cleaning. "
            "Check clean_data.csv values and feature generation."
        )

    return df


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    df = load_data(INPUT_FILE)
    df = force_base_numeric(df)
    df = add_datetime_features(df)
    df = add_trend_and_lags(df)
    df = finalize(df)

    df.to_csv(OUTPUT_FILE, index=False)

    metadata = {
        "input_file": INPUT_FILE,
        "output_file": OUTPUT_FILE,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Feature engineering completed successfully.")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()