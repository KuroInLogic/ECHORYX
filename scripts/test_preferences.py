import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from services.profile_service import recommend_from_preferences
from services.hybrid_service import recommend_hybrid

train = pd.read_parquet("data/processed/train.parquet")
tracks = pd.read_parquet("data/processed/tracks.parquet")

# Select a listener with sufficient history.
user_counts = train.groupby("user_idx").size()
user_idx = int(user_counts[user_counts >= 5].index[0])

# Select artists that exist in the dataset.
selected_artists = (
    tracks["artist_name"]
    .dropna()
    .value_counts()
    .head(3)
    .index.tolist()
)

print("Selected artists:", selected_artists)

# Inspect the preference service independently.
preference_recommendations, preference_stats = (
    recommend_from_preferences(
        selected_artists=selected_artists,
        train=train,
        tracks=tracks,
        n=20
    )
)

print("\nPREFERENCE SERVICE RESULTS:")
print("Statistics:", preference_stats)
print("Recommendations returned:", len(preference_recommendations))

if not preference_recommendations.empty:
    print(
        preference_recommendations[
            ["track_idx", "track_name", "artist_name", "preference_score"]
        ].head(10)
    )

# Test 1: Normal hybrid recommendations.
normal = recommend_hybrid(
    user_idx=user_idx,
    train=train,
    tracks=tracks,
    n=10
)

# Test 2: Preference-enhanced recommendations.
personalised = recommend_hybrid(
    user_idx=user_idx,
    train=train,
    tracks=tracks,
    n=10,
    selected_artists=selected_artists,
    discovery_intensity=0.80
)

assert not normal.empty, "Normal recommendations are empty."
assert not personalised.empty, "Personalised recommendations are empty."

assert "artist_preference_score" in personalised.columns, (
    "Preference score column is missing."
)

assert "hybrid_score" in personalised.columns, (
    "Hybrid score column is missing."
)

print("\nNormal recommendations:")
print(normal[["track_name", "artist_name", "hybrid_score"]].head())

print("\nPersonalised recommendations:")
print(
    personalised[
        ["track_name", "artist_name",
         "artist_preference_score", "hybrid_score"]
    ].head()
)

print("\nPreference integration tests passed!")
