import torch

class PCAProjector():

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def __call__(self, x):
        _, _, V = torch.pca_lowrank(x, q=self.latent_dim)
        x_proj = x @ V
        return x_proj
            