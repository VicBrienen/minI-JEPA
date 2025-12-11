import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.patch_size = patch_size

        # embed using convolution with stride = patch_size
        self.embed = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)
    
    def forward(self, x):           # (B, C, H, W)
        x = self.embed(x)           # (B, E, H, W)
        x = x.flatten(2)            # (B, E, N)
        return x.transpose(1, 2)    # (B, N, E)