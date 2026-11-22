
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

sys.path.insert(0, str(ROOT))

from models.neumf import NeuralCollaborativeFiltering


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_training_data():
    """
    Load training interactions and construct the
    training-observed negative sampling catalogue.

    The test dataset is deliberately NOT loaded.
    """

    train = pd.read_parquet(
        DATA_DIR / "train.parquet",
        columns=["user_idx", "track_idx"]
    )

    tracks = pd.read_parquet(
        DATA_DIR / "tracks.parquet",
        columns=["track_idx"]
    )

    users = train["user_idx"].to_numpy(dtype=np.int64)
    items = train["track_idx"].to_numpy(dtype=np.int64)

    num_users = int(train["user_idx"].max()) + 1
    num_items = int(tracks["track_idx"].max()) + 1

    # All observed training interactions are positive pairs.
    known_pairs = users * num_items + items
    known_pairs.sort()

    # Negative samples must come from the training-observed
    # catalogue only.
    negative_item_pool = np.unique(items)

    return (
        users,
        items,
        known_pairs,
        negative_item_pool,
        num_users,
        num_items
    )


# --------------------------------------------------
# NEGATIVE SAMPLING
# --------------------------------------------------

def sample_negative_items(
    user_ids,
    negative_item_pool,
    num_items,
    known_pairs,
    rng
):
    """
    Generate one negative item per positive interaction.

    Negative items:
    - Must exist in the training-observed catalogue.
    - Must not be a known positive interaction for that user.
    - Are sampled dynamically.
    """

    negative_items = rng.choice(
        negative_item_pool,
        size=len(user_ids),
        replace=True
    )

    while True:

        candidate_keys = user_ids * num_items + negative_items

        positions = np.searchsorted(
            known_pairs,
            candidate_keys
        )

        safe_positions = np.minimum(
            positions,
            len(known_pairs) - 1
        )

        collisions = (
            (positions < len(known_pairs))
            & (known_pairs[safe_positions] == candidate_keys)
        )

        if not collisions.any():
            break

        negative_items[collisions] = rng.choice(
            negative_item_pool,
            size=int(collisions.sum()),
            replace=True
        )

    return negative_items


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

def train_model(args):

    print("\nECHORYX - NEURAL COLLABORATIVE FILTERING")
    print("=" * 55)

    # Reproducibility
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load training data
    (
        users,
        items,
        known_pairs,
        negative_item_pool,
        num_users,
        num_items
    ) = load_training_data()

    print(f"Users: {num_users:,}")
    print(f"Embedding track capacity: {num_items:,}")
    print(f"Positive training interactions: {len(users):,}")
    print(f"Known positive pairs: {len(known_pairs):,}")
    print(
        f"Negative sampling catalogue: "
        f"{len(negative_item_pool):,}"
    )

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    # Initialize model
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        model_type=args.model_type
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    criterion = nn.BCEWithLogitsLoss()

    num_interactions = len(users)

    # --------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------

    for epoch in range(args.epochs):

        model.train()

        shuffled_indices = rng.permutation(num_interactions)

        total_loss = 0.0
        batches = 0

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for start in range(
            0,
            num_interactions,
            args.batch_size
        ):

            batch_indices = shuffled_indices[
                start:start + args.batch_size
            ]

            batch_users = users[batch_indices]
            positive_items = items[batch_indices]

            # Generate negative examples
            negative_items = sample_negative_items(
                batch_users,
                negative_item_pool,
                num_items,
                known_pairs,
                rng
            )

            # Combine positive and negative examples
            batch_user_ids = np.concatenate([
                batch_users,
                batch_users
            ])

            batch_item_ids = np.concatenate([
                positive_items,
                negative_items
            ])

            batch_user_tensor = torch.as_tensor(
                batch_user_ids,
                dtype=torch.long,
                device=device
            )

            batch_item_tensor = torch.as_tensor(
                batch_item_ids,
                dtype=torch.long,
                device=device
            )

            labels = torch.cat([
                torch.ones(
                    len(batch_users),
                    dtype=torch.float32,
                    device=device
                ),
                torch.zeros(
                    len(batch_users),
                    dtype=torch.float32,
                    device=device
                )
            ])

            # Forward and backward propagation
            optimizer.zero_grad(set_to_none=True)

            logits = model(
                batch_user_tensor,
                batch_item_tensor
            )

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            if batches % 100 == 0:
                print(
                    f"  Batch {batches} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / batches

        print(
            f"Epoch {epoch + 1} completed | "
            f"Average loss: {average_loss:.4f}"
        )

    # --------------------------------------------------
    # SAVE MODEL CHECKPOINT
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODEL_DIR / f"ncf_{args.model_type}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": args.embedding_dim,
        "model_type": args.model_type,
        "mlp_layers": [64, 32, 16, 8],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "negative_sampling": "training_observed_catalogue_only"
    }

    torch.save(
        checkpoint,
        model_path
    )

    print("\nTraining completed!")
    print(f"Saved model: {model_path}")


# --------------------------------------------------
# COMMAND LINE ARGUMENTS
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train an NCF model for ECHORYX"
    )

    parser.add_argument(
        "--model-type",
        choices=["gmf", "mlp", "neumf"],
        default="neumf"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    train_model(args)
