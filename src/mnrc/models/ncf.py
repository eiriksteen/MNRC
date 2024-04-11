import torch
import torch.nn as nn

class NeuralMatrixFactorizer(nn.Module):

    def __init__(self, num_users: int, num_items: int, latent_dim: int):
        super().__init__()

        self.user_matrix_gmf = nn.Embedding(num_users, latent_dim)
        self.item_matrix_gmf = nn.Embedding(num_items, latent_dim)
        self.user_matrix_mlp = nn.Embedding(num_users, latent_dim)
        self.item_matrix_mlp = nn.Embedding(num_items, latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2*latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.linear = nn.Linear(2*latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):

        user_vecs_gmf = self.user_matrix_gmf.weight[user_ids].squeeze(dim=1)
        content_vecs_gmf = self.item_matrix_gmf.weight[item_ids].squeeze(dim=1)
        logits_gmf = user_vecs_gmf*content_vecs_gmf

        user_vecs_mlp = self.user_matrix_mlp.weight[user_ids].squeeze(dim=1)
        item_vecs_mlp = self.item_matrix_mlp.weight[item_ids].squeeze(dim=1)
        logits_mlp = self.mlp(torch.cat((user_vecs_mlp, item_vecs_mlp), dim=-1)).squeeze(dim=1)

        logits = self.sigmoid(self.linear(torch.cat((logits_gmf, logits_mlp), dim=-1)))

        return logits