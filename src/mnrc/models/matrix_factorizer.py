import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class MatrixFactorizer(nn.Module):

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
        self.text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.text_encoder.requires_grad_(False)
        self.text_projector = nn.Linear(384, latent_dim) if wtext else None
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

        logits = (user_vecs*(content_vecs+encoded_text)).sum(dim=-1)
        scores = self.sigmoid(logits)

        return logits[:, None], scores[:, None]
    
