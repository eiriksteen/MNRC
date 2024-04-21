import torch
import torch.nn as nn

class NeuralMatrixFactorizer(nn.Module):

    def __init__(
            self, 
            num_users: int, 
            num_items: int, 
            latent_dim: int,
            wtext: bool = False
            ):
        super().__init__()

        self.latent_dim = latent_dim
        self.wtext = wtext

        self.text_projector = nn.Linear(384, latent_dim) if wtext else None
        self.user_matrix_gmf = nn.Embedding(num_users, latent_dim)
        self.item_matrix_gmf = nn.Embedding(num_items, latent_dim)
        self.user_matrix_nl = nn.Embedding(num_users, latent_dim)
        self.item_matrix_nl = nn.Embedding(num_items, latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2*latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.text_mlp = nn.Sequential(
            nn.Linear(3*latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        ) if wtext else nn.Identity()

        self.linear = nn.Linear((3 if wtext else 2)*latent_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, user_ids, item_ids, encoded_text=None):

        user_vecs_gmf = self.user_matrix_gmf.weight[user_ids].squeeze(dim=1)
        item_vecs_gmf = self.item_matrix_gmf.weight[item_ids].squeeze(dim=1)
        user_vecs_nl = self.user_matrix_nl.weight[user_ids].squeeze(dim=1)
        item_vecs_nl = self.item_matrix_nl.weight[item_ids].squeeze(dim=1)

        logits_gmf = user_vecs_gmf*item_vecs_gmf
        logits_nl = self.mlp(torch.cat((user_vecs_nl, item_vecs_nl), dim=-1))

        if encoded_text is not None and self.wtext:
            encoded_text = self.text_projector(encoded_text)
            logits_text = self.text_mlp(torch.cat((user_vecs_gmf, item_vecs_gmf, encoded_text), dim=-1))
        elif self.wtext:
            logits_text = torch.zeros_like(logits_gmf).to(logits_gmf.device)

        if self.wtext:
            logits = self.linear(torch.cat((logits_gmf, logits_nl, logits_text), dim=-1))
        else:
            logits = self.linear(torch.cat((logits_gmf, logits_nl), dim=-1))

        scores = self.sigmoid(logits)
        
        return logits, scores