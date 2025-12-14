import torch
from torch import nn
from custom_modules import PatchEmbedding, TransformerBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim, depth, heads, mlp_ratio):
        super().__init__()

        # patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)       # (B, T, embed_dim)
        self.num_patches = (img_size // patch_size) ** 2

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        pass