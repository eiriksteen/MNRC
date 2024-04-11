import torch.nn as nn

class MatrixFactorizer(nn.Module):

    def __init__(self, num_users: int, num_items: int, latent_dim: int):
        super().__init__()

        self.user_matrix = nn.Embedding(num_users, latent_dim)
        self.item_matrix = nn.Embedding(num_items, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, content_ids):

        user_vecs = self.user_matrix.weight[user_ids]
        content_vecs = self.item_matrix.weight[content_ids]
        logits = self.sigmoid((user_vecs*content_vecs).sum(dim=-1))

        return logits