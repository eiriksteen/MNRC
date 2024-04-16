import torch.nn as nn
from sentence_transformers import SentenceTransformer

class MatrixFactorizer(nn.Module):

    def __init__(self, num_users: int, num_items: int, latent_dim: int):
        super().__init__()

        self.user_matrix = nn.Embedding(num_users, latent_dim)
        self.item_matrix = nn.Embedding(num_items, latent_dim)
        self.embedding_proj = nn.Linear(384, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def init_weights_from_text(self, article_texts):
        print("INITIALIZING ITEM EMBEDDINGS USING TEXT EMBEDDINGS (MIGHT TAKE SOME TIME)")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode(article_texts, convert_to_tensor=True)
        embeddings = self.embedding_proj(embeddings.to(self.embedding_proj.weight.data.device))
        self.item_matrix.weight.data = embeddings
        print("INITIALIZATION DONE")

    def forward(self, user_ids, content_ids):

        user_vecs = self.user_matrix.weight[user_ids]
        content_vecs = self.item_matrix.weight[content_ids]
        logits = self.sigmoid((user_vecs*content_vecs).sum(dim=-1))

        return logits
    
