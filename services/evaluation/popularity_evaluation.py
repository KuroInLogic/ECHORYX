
from pathlib import Path
import pandas as pd

from services.evaluation.metrics import calculate_metrics

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def evaluate_popularity_model(k=10):

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    ranking = pd.read_parquet(DATA_DIR / "popularity_ranking.parquet")

    ranked_tracks = ranking["track_idx"].tolist()
    ranked_track_set = set(ranked_tracks)

    train_history = (
        train.groupby("user_idx")["track_idx"]
        .apply(set)
        .to_dict()
    )

    test_history = (
        test.groupby("user_idx")["track_idx"]
        .apply(set)
        .to_dict()
    )

    recommendations_by_user = {}

    cold_start_tracks = set()
    cold_start_test_interactions = 0

    for user_idx, relevant_tracks in test_history.items():

        listened_tracks = train_history.get(user_idx, set())

        recommendations = []

        for track in ranked_tracks:

            if track not in listened_tracks:
                recommendations.append(track)

            if len(recommendations) == k:
                break

        recommendations_by_user[user_idx] = recommendations

        unavailable_tracks = relevant_tracks - ranked_track_set

        cold_start_tracks.update(unavailable_tracks)
        cold_start_test_interactions += len(unavailable_tracks)

    results = calculate_metrics(
        recommendations_by_user,
        test_history,
        k
    )

    results.update({
        "cold_start_unique_tracks": len(cold_start_tracks),
        "cold_start_test_interactions": cold_start_test_interactions
    })

    return results
