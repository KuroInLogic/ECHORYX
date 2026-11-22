from pathlib import Path
import math
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def evaluate_popularity_model(k=10):
    """Evaluate popularity recommendations against held-out interactions."""

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

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    total_hits = 0
    total_relevant = 0

    cold_start_tracks = set()
    recommended_track_set = set()
    cold_start_test_interactions = 0

    for user_idx, relevant_tracks in test_history.items():

        listened_tracks = train_history.get(user_idx, set())

        recommendations = []

        for track in ranked_tracks:
            if track not in listened_tracks:
                recommendations.append(track)

            if len(recommendations) == k:
                break

        recommended_track_set.update(recommendations)

        hits = [
            track for track in recommendations
            if track in relevant_tracks
        ]

        hit_count = len(hits)

        precision = hit_count / k
        recall = hit_count / len(relevant_tracks)

        dcg = sum(
            1 / math.log2(position + 2)
            for position, track in enumerate(recommendations)
            if track in relevant_tracks
        )

        ideal_length = min(k, len(relevant_tracks))

        idcg = sum(
            1 / math.log2(position + 2)
            for position in range(ideal_length)
        )

        ndcg = dcg / idcg if idcg else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

        total_hits += hit_count
        total_relevant += len(relevant_tracks)

        unavailable_tracks = relevant_tracks - ranked_track_set

        cold_start_tracks.update(unavailable_tracks)
        cold_start_test_interactions += len(unavailable_tracks)

    user_count = len(test_history)

    return {
        "users_evaluated": user_count,
        "precision_at_k": sum(precision_scores) / user_count,
        "recall_at_k": sum(recall_scores) / user_count,
        "ndcg_at_k": sum(ndcg_scores) / user_count,
        "total_hits": total_hits,
        "total_test_tracks": total_relevant,
        "cold_start_unique_tracks": len(cold_start_tracks),
        "cold_start_test_interactions": cold_start_test_interactions,
        "unique_recommended_tracks": len(recommended_track_set),
        "k": k
    }