
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from services.ncf_service import load_ncf_model
from services.evaluation.metrics import calculate_metrics


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


def evaluate_ncf_model(
    k=10,
    batch_size=8192,
    max_users=None
):
    """Evaluate NeuMF using full-catalogue ranking."""

    train = pd.read_parquet(
        DATA_DIR / "train.parquet",
        columns=["user_idx", "track_idx"]
    )

    test = pd.read_parquet(
        DATA_DIR / "test.parquet",
        columns=["user_idx", "track_idx"]
    )

    # Restrict recommendations to tracks observed in training.
    all_track_ids = np.sort(
        train["track_idx"].unique().astype(np.int64)
    )

    model, device = load_ncf_model()

    # Build each user's training history.
    train_history = (
        train.groupby("user_idx")["track_idx"]
        .apply(set)
        .to_dict()
    )

    # Build each user's held-out test history.
    test_history = (
        test.groupby("user_idx")["track_idx"]
        .apply(set)
        .to_dict()
    )

    users = sorted(test_history.keys())

    if max_users is not None:
        users = users[:max_users]

    recommendations_by_user = {}

    print("\nEvaluating NeuMF")
    print("=" * 45)
    print("Device:", device)
    print("Users to evaluate:", len(users))
    print("Candidate catalogue:", len(all_track_ids))

    model.eval()

    with torch.inference_mode():

        for position, user_idx in enumerate(users, start=1):

            seen = train_history.get(user_idx, set())

            candidates = np.array(
                [
                    track_id
                    for track_id in all_track_ids
                    if track_id not in seen
                ],
                dtype=np.int64
            )

            candidate_scores = np.empty(
                len(candidates),
                dtype=np.float32
            )

            for start in range(0, len(candidates), batch_size):

                end = min(start + batch_size, len(candidates))

                item_batch = candidates[start:end]

                user_batch = torch.full(
                    (len(item_batch),),
                    int(user_idx),
                    dtype=torch.long,
                    device=device
                )

                item_tensor = torch.as_tensor(
                    item_batch,
                    dtype=torch.long,
                    device=device
                )

                # Use logits for ranking. Sigmoid is monotonic,
                # so applying it would not change the ranking.
                logits = model(user_batch, item_tensor)

                candidate_scores[start:end] = (
                    logits.cpu().numpy()
                )

            top_count = min(k, len(candidates))

            top_positions = np.argpartition(
                candidate_scores, -top_count
            )[-top_count:]

            top_positions = top_positions[
                np.argsort(
                    candidate_scores[top_positions]
                )[::-1]
            ]

            recommendations_by_user[user_idx] = (
                candidates[top_positions].tolist()
            )

            if position % 10 == 0 or position == len(users):
                print(
                    f"Processed {position}/{len(users)} users"
                )

    # Evaluate only users for whom recommendations were generated.
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
