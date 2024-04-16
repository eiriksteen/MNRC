import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class MatrixFactorizer(nn.Module):

    def __init__(self, num_users: int, num_items: int, latent_dim: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.user_matrix = nn.Embedding(num_users, latent_dim)
        self.item_matrix = nn.Embedding(num_items, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def init_weights_from_text_pca(self, article_texts):
        print("INITIALIZING ITEM EMBEDDINGS USING TEXT EMBEDDINGS (MIGHT TAKE SOME TIME)")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        device = self.item_matrix.weight.data.device
        embeddings = model.encode(article_texts, convert_to_tensor=True).to(device)
        _, _, V = torch.pca_lowrank(embeddings, q=self.latent_dim)
        embeddings = embeddings @ V
        self.item_matrix.weight.data = embeddings
        print("INITIALIZATION DONE")

    def forward(self, user_ids, content_ids):

        user_vecs = self.user_matrix.weight[user_ids]
        content_vecs = self.item_matrix.weight[content_ids]
        logits = self.sigmoid((user_vecs*content_vecs).sum(dim=-1))

        return logits
    
