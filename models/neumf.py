
import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering implementation.

    Supported architectures:
    - GMF: Generalized Matrix Factorization
    - MLP: Multi-Layer Perceptron
    - NeuMF: Combination of GMF and MLP

    The forward method returns logits.
    Use BCEWithLogitsLoss during training.
    Apply sigmoid to logits when probabilities are required.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=32,
        model_type="neumf",
        mlp_layers=None
    ):
        super().__init__()

        if model_type not in {"gmf", "mlp", "neumf"}:
            raise ValueError(
                "model_type must be 'gmf', 'mlp', or 'neumf'"
            )

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.model_type = model_type

        if mlp_layers is None:
            mlp_layers = [64, 32, 16, 8]

        # GMF embeddings
        if model_type in {"gmf", "neumf"}:
            self.gmf_user_embedding = nn.Embedding(
                num_users, embedding_dim
            )
            self.gmf_item_embedding = nn.Embedding(
                num_items, embedding_dim
            )

        # MLP embeddings and layers
        if model_type in {"mlp", "neumf"}:
            self.mlp_user_embedding = nn.Embedding(
                num_users, embedding_dim
            )
            self.mlp_item_embedding = nn.Embedding(
                num_items, embedding_dim
            )

            layers = []
            input_size = embedding_dim * 2

            for output_size in mlp_layers:
                layers.append(nn.Linear(input_size, output_size))
                layers.append(nn.ReLU())
                input_size = output_size

            self.mlp_network = nn.Sequential(*layers)

        # Final prediction layer
        if model_type == "gmf":
            prediction_input_size = embedding_dim

        elif model_type == "mlp":
            prediction_input_size = mlp_layers[-1]

        else:
            prediction_input_size = (
                embedding_dim + mlp_layers[-1]
            )

        self.output_layer = nn.Linear(
            prediction_input_size, 1
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize embeddings and neural network weights."""

        for module in self.modules():

            if isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, mean=0.0, std=0.01
                )

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_ids, item_ids):

        representations = []

        # GMF branch
        if self.model_type in {"gmf", "neumf"}:

            user_gmf = self.gmf_user_embedding(user_ids)
            item_gmf = self.gmf_item_embedding(item_ids)

            gmf_vector = user_gmf * item_gmf

            if self.model_type == "gmf":
                representations.append(gmf_vector)

        # MLP branch
        if self.model_type in {"mlp", "neumf"}:

            user_mlp = self.mlp_user_embedding(user_ids)
            item_mlp = self.mlp_item_embedding(item_ids)

            mlp_vector = torch.cat(
                [user_mlp, item_mlp], dim=-1
            )

            mlp_vector = self.mlp_network(mlp_vector)

            representations.append(mlp_vector)

        # Combine branches for NeuMF
        if self.model_type == "neumf":

            representations = [
                gmf_vector,
                mlp_vector
            ]

        final_vector = torch.cat(
            representations, dim=-1
        ) if len(representations) > 1 else representations[0]

        logits = self.output_layer(final_vector).squeeze(-1)

        return logits
