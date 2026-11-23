import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from services.hybrid_service import recommend_hybrid

# Load processed data
train = pd.read_parquet("data/processed/train.parquet")
tracks = pd.read_parquet("data/processed/tracks.parquet")

# ---------------------------------------
# TEST 1: New listener with no history
# ---------------------------------------

print("\nTEST 1: Cold-start recommendation")

cold_user = -1

cold_recommendations = recommend_hybrid(
    user_idx=cold_user,
    train=train,
    tracks=tracks,
    n=10
)

assert not cold_recommendations.empty, (
    "Cold-start recommendations are empty."
)

assert cold_recommendations["fallback_used"].all(), (
    "Cold-start fallback was not activated."
)

print("PASS: Popularity fallback activated.")
print(cold_recommendations[
    ["track_name", "artist_name", "hybrid_score"]
].head())


# ---------------------------------------
# TEST 2: Existing listener
# ---------------------------------------

print("\nTEST 2: Existing listener")

user_counts = train.groupby("user_idx").size()

existing_user = int(user_counts[user_counts >= 5].index[0])

normal_recommendations = recommend_hybrid(
    user_idx=existing_user,
    train=train,
    tracks=tracks,
    n=10
)

assert not normal_recommendations.empty, (
    "Normal recommendations are empty."
)

assert not normal_recommendations["fallback_used"].any(), (
    "Unexpected fallback for an established listener."
)

print("PASS: Normal hybrid recommendation activated.")
print(normal_recommendations[
    ["track_name", "artist_name", "hybrid_score"]
].head())

print("\nBoth cold-start tests completed successfully.")
