
import pandas as pd

from services.user_cf_service import recommend_for_user

# Load processed datasets
train = pd.read_parquet("data/processed/train.parquet")
tracks = pd.read_parquet("data/processed/tracks.parquet")

# Select five different listeners
users = train["user_idx"].drop_duplicates().head(5).tolist()

results = {}

for user in users:
    recommendations = recommend_for_user(
        user_idx=user,
        train=train,
        tracks=tracks,
        n=10
    )

    results[user] = set(recommendations["track_idx"])

    print(f"\n{'=' * 45}")
    print(f"USER: {user}")
    print("=" * 45)

    if recommendations.empty:
        print("No recommendations found.")
        continue

    print(
        recommendations[
            ["artist_name", "track_name", "listener_count"]
        ].head(5).to_string(index=False)
    )

# Compare recommendation diversity
if results:
    unique_tracks = set.union(*results.values())

    print("\n" + "=" * 45)
    print("COMPARISON")
    print("=" * 45)

    print("Users tested:", len(users))
    print("Unique recommended tracks:", len(unique_tracks))
