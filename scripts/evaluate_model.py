
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.evaluation.popularity_evaluation import (
    evaluate_popularity_model
)

from services.evaluation.user_cf_evaluation import (
    evaluate_user_cf_model
)


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("             ECHORYX MODEL EVALUATION")
    print("=" * 55)

    print("\n[1/2] Evaluating Popularity-Based Model...")

    popularity = evaluate_popularity_model(k=10)

    print("\n[2/2] Evaluating User-Based Collaborative Filtering...")

    user_cf = evaluate_user_cf_model(
        k=10,
        similar_user_limit=50
    )

    print("\n" + "=" * 55)
    print("                 RESULTS COMPARISON")
    print("=" * 55)

    print(
        f"\n{'Metric':<25}"
        f"{'Popularity':>14}"
        f"{'User-Based CF':>16}"
    )

    print("-" * 55)

    metrics = [
        ("Precision@10", "precision_at_k"),
        ("Recall@10", "recall_at_k"),
        ("NDCG@10", "ndcg_at_k")
    ]

    for label, key in metrics:
        print(
            f"{label:<25}"
            f"{popularity[key]:>14.4f}"
            f"{user_cf[key]:>16.4f}"
        )

    print("\n" + "=" * 55)
    print("              ADDITIONAL STATISTICS")
    print("=" * 55)

    print("\nUsers evaluated:", popularity["users_evaluated"])

    print(
        "Popularity recommendation hits:",
        popularity["total_hits"]
    )

    print(
        "User-CF recommendation hits:",
        user_cf["total_hits"]
    )

    print(
        "Popularity unique tracks:",
        popularity["unique_recommended_tracks"]
    )

    print(
        "User-CF unique tracks:",
        user_cf["unique_recommended_tracks"]
    )

    print("\nEvaluation completed successfully.")
