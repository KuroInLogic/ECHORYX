
import pandas as pd

from services.hybrid_service import recommend_hybrid

train = pd.read_parquet("data/processed/train.parquet")
tracks = pd.read_parquet("data/processed/tracks.parquet")

user_idx = int(train["user_idx"].iloc[0])

from services.hybrid_service import _get_popularity_recommendations

pop_test = _get_popularity_recommendations(
    user_idx=user_idx,
    train=train,
    n=200
)

print("\nPOPULARITY TEST")
print("Number of recommendations:", len(pop_test))
print(pop_test.head(5).to_string(index=False))

recommendations = recommend_hybrid(
    user_idx=user_idx,
    train=train,
    tracks=tracks,
    n=10
)

print("\nHYBRID RECOMMENDATIONS")
print(
    recommendations[
        [
            "track_name",
            "artist_name",
            "hybrid_score",
            "user_cf_score",
            "ncf_score",
            "popularity_score"
        ]
    ].to_string(index=False)
)
