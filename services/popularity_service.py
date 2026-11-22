
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def build_popularity_ranking():
    """Build a global popularity ranking using training data only."""

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    tracks = pd.read_parquet(DATA_DIR / "tracks.parquet")

    popularity = (
        train.groupby("track_idx")
        .agg(
            listener_count=("user_idx", "nunique"),
            total_plays=("play_count", "sum")
        )
        .reset_index()
    )

    popularity = popularity.merge(
        tracks[["track_idx", "artist_name", "track_name"]],
        on="track_idx",
        how="left"
    )

    popularity = popularity.sort_values(
        ["listener_count", "total_plays", "track_idx"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    popularity.to_parquet(
        DATA_DIR / "popularity_ranking.parquet",
        index=False
    )

    return popularity


def recommend_for_user(user_idx, n=10):
    """Recommend popular tracks the user has not heard in training."""

    ranking_path = DATA_DIR / "popularity_ranking.parquet"

    if not ranking_path.exists():
        raise FileNotFoundError(
            "Popularity ranking not found. Build the model first."
        )

    ranking = pd.read_parquet(ranking_path)
    train = pd.read_parquet(DATA_DIR / "train.parquet")

    listened_tracks = set(
        train.loc[
            train["user_idx"] == user_idx,
            "track_idx"
        ]
    )

    recommendations = ranking[
        ~ranking["track_idx"].isin(listened_tracks)
    ].head(n)

    return recommendations.reset_index(drop=True)
