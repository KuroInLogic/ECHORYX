
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from services.evaluation.metrics import calculate_metrics

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def evaluate_user_cf_model(k=10, similar_user_limit=50):

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    user_ids = sorted(train["user_idx"].unique())
    track_ids = sorted(train["track_idx"].unique())

    user_to_pos = {
        user_id: position
        for position, user_id in enumerate(user_ids)
    }

    track_to_pos = {
        track_id: position
        for position, track_id in enumerate(track_ids)
    }

    row = train["user_idx"].map(user_to_pos).to_numpy()
    col = train["track_idx"].map(track_to_pos).to_numpy()

    binary_matrix = csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (row, col)
        ),
        shape=(len(user_ids), len(track_ids))
    )

    play_matrix = csr_matrix(
        (
            train["play_count"].to_numpy(dtype=np.float32),
            (row, col)
        ),
        shape=(len(user_ids), len(track_ids))
    )

    user_norms = np.sqrt(
        np.asarray(
            binary_matrix.multiply(binary_matrix).sum(axis=1)
        ).ravel()
    )

    similarity_matrix = (
        binary_matrix @ binary_matrix.T
    ).toarray()

    denominators = np.outer(user_norms, user_norms)

    similarity_matrix = np.divide(
        similarity_matrix,
        denominators,
        out=np.zeros_like(
            similarity_matrix,
            dtype=np.float32
        ),
        where=denominators != 0
    )

    test_history = (
        test.groupby("user_idx")["track_idx"]
        .apply(set)
        .to_dict()
    )

    recommendations_by_user = {}

    print("Evaluating User-Based CF...")

    for count, (user_idx, _) in enumerate(
        test_history.items(), start=1
    ):

        if user_idx not in user_to_pos:
            recommendations_by_user[user_idx] = []
            continue

        target_pos = user_to_pos[user_idx]

        user_similarities = similarity_matrix[target_pos].copy()
        user_similarities[target_pos] = 0

        similar_positions = np.argsort(
            user_similarities
        )[::-1][:similar_user_limit]

        similar_positions = [
            pos for pos in similar_positions
            if user_similarities[pos] > 0
        ]

        if not similar_positions:
            recommendations_by_user[user_idx] = []
            continue

        weights = user_similarities[similar_positions]

        score_vector = np.asarray(
            weights.reshape(1, -1)
            @ play_matrix[similar_positions]
        ).ravel()

        listener_vector = np.asarray(
            binary_matrix[similar_positions].sum(axis=0)
        ).ravel()

        total_plays_vector = np.asarray(
            play_matrix[similar_positions].sum(axis=0)
        ).ravel()

        target_history = binary_matrix[target_pos].indices

        score_vector[target_history] = 0

        candidate_positions = np.flatnonzero(score_vector > 0)

        if len(candidate_positions) == 0:
            recommendations_by_user[user_idx] = []
            continue

        order = np.lexsort((
            -total_plays_vector[candidate_positions],
            -listener_vector[candidate_positions],
            -score_vector[candidate_positions]
        ))

        top_positions = candidate_positions[order[:k]]

        recommendations_by_user[user_idx] = [
            track_ids[pos] for pos in top_positions
        ]

        if count % 100 == 0:
            print(
                f"Processed {count}/{len(test_history)} users"
            )

    return calculate_metrics(
        recommendations_by_user,
        test_history,
        k
    )
