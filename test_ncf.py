
import torch
from models.neumf import NeuralCollaborativeFiltering

NUM_USERS = 951
NUM_TRACKS = 175546
BATCH_SIZE = 32

for model_type in ["gmf", "mlp", "neumf"]:

    model = NeuralCollaborativeFiltering(
        num_users=NUM_USERS,
        num_items=NUM_TRACKS,
        embedding_dim=32,
        model_type=model_type
    )

    user_ids = torch.randint(0, NUM_USERS, (BATCH_SIZE,))
    track_ids = torch.randint(0, NUM_TRACKS, (BATCH_SIZE,))

    logits = model(user_ids, track_ids)
    probabilities = torch.sigmoid(logits)

    print(f"\nModel: {model_type.upper()}")
    print("Output shape:", logits.shape)
    print("Probability range:",
          probabilities.min().item(),
          "-",
          probabilities.max().item())

    assert logits.shape == (BATCH_SIZE,)
    assert torch.isfinite(logits).all()

print("\nAll NCF architecture tests passed!")
