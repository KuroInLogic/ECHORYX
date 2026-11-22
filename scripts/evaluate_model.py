import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.evaluation_service import evaluate_popularity_model


if __name__ == "__main__":
    print("\nECHORYX MODEL EVALUATION:\n")

    results = evaluate_popularity_model(k=10)

    print(f"Users evaluated: {results['users_evaluated']}")

    print(f"\nPrecision@{results['k']}: {results['precision_at_k']:.4f}")
    print(f"Recall@{results['k']}: {results['recall_at_k']:.4f}")
    print(f"NDCG@{results['k']}: {results['ndcg_at_k']:.4f}")

    print("\nADDITIONAL STATISTICS:\n")

    print(f"Total recommendation hits: {results['total_hits']}")
    print(f"Total test interactions: {results['total_test_tracks']}")
    print(f"Unique tracks recommended: {results['unique_recommended_tracks']}")
    print(f"Unique cold-start tracks: {results['cold_start_unique_tracks']}")
    print(f"Cold-start test interactions: {results['cold_start_test_interactions']}")

    print("\nEvaluation completed successfully.")
