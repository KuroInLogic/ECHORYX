
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.neumf import NeuralCollaborativeFiltering


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "ncf_neumf.pt"


@lru_cache(maxsize=2)
def _load_ncf_model_cached(model_path_string):
    """Load a trained NCF checkpoint once per model path."""

    model_path = Path(model_path_string)

    if not model_path.exists():
        raise FileNotFoundError(
            f"NCF checkpoint not found: {model_path}. "
            "Train the model first using scripts/train_ncf.py."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model = NeuralCollaborativeFiltering(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
        model_type=checkpoint["model_type"],
        mlp_layers=checkpoint["mlp_layers"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def load_ncf_model(model_path=MODEL_PATH):
    """Return the cached trained model and its device."""

    return _load_ncf_model_cached(
        str(Path(model_path).resolve())
    )


def recommend_for_user(
    user_idx,
    train,
    tracks,
    n=10,
    model_path=MODEL_PATH,
    batch_size=4096
):
    """
    Generate Top-N NeuMF recommendations for an existing user.

    Candidate tracks are restricted to tracks observed in the
    training interactions, matching the offline evaluation setup.

    Tracks already heard by the user in training are excluded.
    """

    model, device = load_ncf_model(model_path)

    user_idx = int(user_idx)

    if user_idx < 0 or user_idx >= model.num_users:
        raise ValueError("Invalid user_idx")

    if n <= 0:
        raise ValueError("n must be greater than zero")

    required_train_columns = {"user_idx", "track_idx"}

    if not required_train_columns.issubset(train.columns):
        raise ValueError(
            "train must contain user_idx and track_idx columns."
        )

    required_track_columns = {
        "track_idx",
        "track_name",
        "artist_name"
    }

    if not required_track_columns.issubset(tracks.columns):
        raise ValueError(
            "tracks must contain track_idx, track_name and artist_name."
        )

    # Identify tracks already heard by this user.
    user_history = train.loc[
        train["user_idx"] == user_idx,
        "track_idx"
    ].astype(np.int64)

    seen_tracks = set(user_history.tolist())

    # Match the evaluation candidate catalogue:
    # only tracks that appeared in training interactions.
    candidate_catalogue = np.sort(
        train["track_idx"]
        .dropna()
        .unique()
        .astype(np.int64)
    )

    candidate_ids = np.array(
        [
            track_id
            for track_id in candidate_catalogue
            if track_id not in seen_tracks
            and 0 <= track_id < model.num_items
        ],
        dtype=np.int64
    )

    if len(candidate_ids) == 0:
        return pd.DataFrame(
            columns=[
                "track_idx",
                "preference_score",
                "track_name",
                "artist_name"
            ]
        )

    scores = np.empty(
        len(candidate_ids),
        dtype=np.float32
    )

    # Score candidates in batches to control memory usage.
    with torch.no_grad():

        for start in range(
            0,
            len(candidate_ids),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(candidate_ids)
            )

            item_batch = candidate_ids[start:end]

            user_batch = torch.full(
                (len(item_batch),),
                user_idx,
                dtype=torch.long,
                device=device
            )

            item_tensor = torch.as_tensor(
                item_batch,
                dtype=torch.long,
                device=device
            )

            logits = model(
                user_batch,
                item_tensor
            )

            probabilities = torch.sigmoid(logits)

            scores[start:end] = (
                probabilities.cpu().numpy()
            )

    # Select the highest-scoring candidates.
    top_count = min(n, len(scores))

    top_positions = np.argpartition(
        scores,
        -top_count
    )[-top_count:]

    top_positions = top_positions[
        np.argsort(scores[top_positions])[::-1]
    ]

    recommendations = pd.DataFrame({
        "track_idx": candidate_ids[top_positions],
        "preference_score": scores[top_positions]
    })

    # Attach track and artist metadata.
    track_metadata = tracks[
        ["track_idx", "track_name", "artist_name"]
    ].drop_duplicates(
        subset=["track_idx"]
    )

    recommendations = recommendations.merge(
        track_metadata,
        on="track_idx",
        how="left"
    )

    return recommendations.reset_index(drop=True)
