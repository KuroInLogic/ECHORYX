
import pandas as pd

from services.ncf_service import recommend_for_user


train = pd.read_parquet(
    "data/processed/train.parquet"
)

tracks = pd.read_parquet(
    "data/processed/tracks.parquet"
)

# Test using the first existing user.
user_idx = 0

recommendations = recommend_for_user(
    user_idx=user_idx,
    train=train,
    tracks=tracks,
    n=10
)

seen_tracks = set(
    train.loc[
        train["user_idx"] == user_idx,
        "track_idx"
    ]
)

print("\nECHORYX - NEUMF RECOMMENDATIONS")
print("=" * 60)

print(recommendations.to_string(index=False))

print("\nValidation:")
print("Recommendation count:", len(recommendations))
print(
    "Previously heard tracks:",
    recommendations["track_idx"].isin(seen_tracks).sum()
)

assert len(recommendations) == 10
assert recommendations["track_idx"].is_unique
assert not recommendations["track_idx"].isin(seen_tracks).any()
assert recommendations["track_name"].notna().all()

print("\nAll recommendation tests passed!")
