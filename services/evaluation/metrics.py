
import math


def calculate_metrics(recommendations_by_user, test_history, k=10):
    """Calculate Precision@K, Recall@K and NDCG@K."""

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    total_hits = 0
    total_relevant = 0
    recommended_tracks = set()

    for user_idx, relevant_tracks in test_history.items():

        recommendations = recommendations_by_user.get(user_idx, [])

        recommended_tracks.update(recommendations)

        hits = [
            track for track in recommendations
            if track in relevant_tracks
        ]

        hit_count = len(hits)

        precision_scores.append(hit_count / k)

        recall_scores.append(
            hit_count / len(relevant_tracks)
            if relevant_tracks else 0
        )

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

        ndcg_scores.append(dcg / idcg if idcg else 0)

        total_hits += hit_count
        total_relevant += len(relevant_tracks)

    user_count = len(test_history)

    return {
        "users_evaluated": user_count,
        "precision_at_k": sum(precision_scores) / user_count if user_count else 0,
        "recall_at_k": sum(recall_scores) / user_count if user_count else 0,
        "ndcg_at_k": sum(ndcg_scores) / user_count if user_count else 0,
        "total_hits": total_hits,
        "total_test_tracks": total_relevant,
        "unique_recommended_tracks": len(recommended_tracks),
        "k": k
    }
