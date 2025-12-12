import torch
from torch import nn
from custom_modules import PatchEmbedding, TransformerBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim, depth, heads, mlp_ratio):
        super().__init__()

        # patch embedding + positional encoding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)       # (B, T, embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))  # (B, T, embed_dim)

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)
        ])

    def forward(self, x):                           # (B, C, H, W)
        x = self.patch_embed(x) + self.pos_embed    # (B, T, embed_dim)
        x = self.blocks(x)
        return self.norm(x)                         # (B, T, embed_dim)

class Predicator(nn.Module):
    pass 

class JEPA(nn.Module):
    pass