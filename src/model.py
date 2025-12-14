import torch
from torch import nn
from custom_modules import PatchEmbedding, TransformerBlock
from helper import get_2d_pos_embed

class Encoder(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim, depth, heads, mlp_ratio):
        super().__init__()

        # patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)       # (B, T, embed_dim)

        # 2d sin-cos positional embedding
        grid_size = img_size // patch_size
        pos_embed = get_2d_pos_embed(embed_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0)) # (1, T, embed_dim)

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # patch embedding + positional embedding
        x = self.patch_embed(x) + self.pos_embed

        # pass trough transformer blocks
        for block in self.blocks:
            x = block(x)

        # normalize and return
        return self.norm(x)