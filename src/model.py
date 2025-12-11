from torch import nn
from custom_modules import PatchEmbedding

class Encoder(nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, emb_dim, patch_size)

    def forward(self, x):               # (B, C, H, W)
        x = self.patch_embedding(x)     # (B, N, E)

class Predicator(nn.Module):
    pass 

class JEPA(nn.Module):
    pass