
from pathlib import Path

import numpy as np
import pandas as pd

from services.hybrid_service import recommend_hybrid
from services.evaluation.metrics import calculate_metrics


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


def evaluate_hybrid_model(
    k=10,
    max_users=None,
    weights=None
):
    """
    Evaluate Hybrid recommendations using the same
    train/test split and candidate catalogue as other models.
    """

    train = pd.read_parquet(
        DATA_DIR / "train.parquet"
    )

    test = pd.read_parquet(
        DATA_DIR / "test.parquet",
        columns=["user_idx", "track_idx"]
    )

    tracks = pd.read_parquet(
        DATA_DIR / "tracks.parquet",
        columns=["track_idx", "track_name", "artist_name"]
    )

    # Candidate catalogue: tracks observed in training.
    all_track_ids = np.sort(
        train["track_idx"].unique().astype(np.int64)
    )

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

    users = sorted(test_history.keys())

    if max_users is not None:
        users = users[:max_users]

    recommendations_by_user = {}

    print("\nEvaluating Hybrid Recommendation Model")
    print("=" * 50)
    print("Users to evaluate:", len(users))
    print("Candidate catalogue:", len(all_track_ids))
    print("Weights:", weights or {
        "user_cf": 0.40,
        "ncf": 0.40,
        "popularity": 0.20
    })

    for position, user_idx in enumerate(users, start=1):

        recommendations = recommend_hybrid(
            user_idx=int(user_idx),
            train=train,
            tracks=tracks,
            n=k,
            candidate_pool=len(all_track_ids),
            weights=weights
        )

        recommendations_by_user[user_idx] = (
            recommendations["track_idx"].astype(int).tolist()
        )

        if position % 10 == 0 or position == len(users):
            print(
                f"Processed {position}/{len(users)} users"
            )

    evaluation_test_history = {
        user_idx: test_history[user_idx]
        for user_idx in recommendations_by_user
    }

    results = calculate_metrics(
        recommendations_by_user,
        evaluation_test_history,
        k=k
    )

    return results


if __name__ == "__main__":

    results = evaluate_hybrid_model(
        k=10,
        max_users=None
    )

    print("\nHYBRID EVALUATION RESULTS")
    print("=" * 40)

    for metric, value in results.items():
        print(f"{metric}: {value}")
