
from pathlib import Path
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def create_temporal_split(test_ratio=0.2):
    """Split each user's interactions chronologically."""

    input_path = DATA_DIR / "interactions.parquet"

    interactions = pd.read_parquet(input_path)

    # Convert timestamps into proper datetime values.
    interactions["last_timestamp"] = pd.to_datetime(
        interactions["last_timestamp"],
        utc=True,
        errors="coerce"
    )

    if interactions["last_timestamp"].isna().any():
        raise ValueError("Invalid timestamps found in the dataset.")

    # Sort each user's interactions chronologically.
    interactions = interactions.sort_values(
        ["user_idx", "last_timestamp", "track_idx"]
    ).reset_index(drop=True)

    # Calculate each user's position and total interaction count.
    interactions["position"] = (
        interactions.groupby("user_idx").cumcount()
    )

    interactions["user_total"] = (
        interactions.groupby("user_idx")["track_idx"].transform("size")
    )

    # Reserve the latest 20% for testing.
    interactions["test_start"] = (
        interactions["user_total"]
        - (interactions["user_total"] * test_ratio).apply(math.ceil)
    )

    train = interactions[
        interactions["position"] < interactions["test_start"]
    ].copy()

    test = interactions[
        interactions["position"] >= interactions["test_start"]
    ].copy()

    # Remove temporary columns.
    temporary_columns = ["position", "user_total", "test_start"]

    train = train.drop(columns=temporary_columns)
    test = test.drop(columns=temporary_columns)

    # Save the datasets.
    train.to_parquet(DATA_DIR / "train.parquet", index=False)
    test.to_parquet(DATA_DIR / "test.parquet", index=False)

    return {
        "total_interactions": len(interactions),
        "training_interactions": len(train),
        "testing_interactions": len(test),
        "total_users": interactions["user_idx"].nunique(),
        "training_users": train["user_idx"].nunique(),
        "testing_users": test["user_idx"].nunique(),
        "training_tracks": train["track_idx"].nunique(),
        "testing_tracks": test["track_idx"].nunique()
    }
