import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.embed = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)
    
    def forward(self, x):
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return 