from pathlib import Path
import json

import pandas as pd

from forecast_features import build_engineered_frame


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


def build_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical engineering: delegates to forecast_features.build_engineered_frame."""

    return build_engineered_frame(df).df


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    df = load_data(INPUT_FILE)
    df = build_engineered(df)

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