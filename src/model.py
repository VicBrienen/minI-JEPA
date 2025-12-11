import torch
from torch import nn
from custom_modules import PatchEmbedding

class Encoder(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, img_size):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.num_patches = (img_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim)) # (B, N, E) -> positional embedding for each patch and same size as embed_dim

    def forward(self, x):           # (B, C, H, W)
        x = self.patch_embed(x)     # (B, N, E)
        x = x + self.pos_embed      # (B, N, E)

class Predicator(nn.Module):
    pass 

class JEPA(nn.Module):
    pass