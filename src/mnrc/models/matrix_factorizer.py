import torch
import torch.nn as nn
from .pca_projector import PCAProjector
from sentence_transformers import SentenceTransformer

class MatrixFactorizer(nn.Module):

    def __init__(
            self, 
            num_users: int, 
            num_items: int, 
            latent_dim: int,
            learn_projection: bool = True
            ):
        super().__init__()

        self.latent_dim = latent_dim
        self.text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.text_encoder.requires_grad_(False)
        
        if learn_projection:
            self.text_projector = nn.Linear(384, latent_dim)
        else:
            self.text_projector = PCAProjector(latent_dim)

        self.user_matrix = nn.Embedding(num_users, latent_dim)
        self.item_matrix = nn.Embedding(num_items, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, content_ids, encoded_text=None):

        user_vecs = self.user_matrix.weight[user_ids].squeeze(dim=1)
        content_vecs = self.item_matrix.weight[content_ids].squeeze(dim=1)

        if encoded_text is not None:
            encoded_text = self.text_projector(encoded_text)
        else:
            encoded_text = torch.zeros_like(content_vecs).to(content_ids.device)

        logits = self.sigmoid((user_vecs*(content_vecs+encoded_text)).sum(dim=-1))

        return logits[:, None]
    
