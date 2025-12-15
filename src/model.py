import torch
from torch import nn
from custom_modules import PatchEmbedding, TransformerBlock
from helper import get_2d_pos_embed, gather_pos_embed

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
    
class Predictor(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, num_patches, heads, mlp_ratio, depth):
        super().__init__()

        # encoder_dim > predictor_dim so that learning representations is forced into the encoder
        self.predictor_embed = nn.Linear(encoder_dim, predictor_dim) # (B, T, predictor_dim)

        # learned mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # 2d sin-cos positional embedding
        grid_size = int(num_patches ** 0.5)
        pos_embed = get_2d_pos_embed(predictor_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0)) # (1, T, predictor_dim)

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, heads, mlp_ratio) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # project back to size of encoder output
        self.proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, x_visible, visible_idx, masked_idx):
        B, _, _ = x_visible.shape

        # reduce embedding dimension for predictor
        x = self.predictor_embed(x_visible)

        # add positional embedding to visible tokens
        pos_visible = gather_pos_embed(self.pos_embed, visible_idx)
        x = x + pos_visible

        # create mask tokens and add positional embedding
        mask_tokens = self.mask_token.expand(B, masked_idx.shape[1], -1)
        pos_masked = gather_pos_embed(self.pos_embed, masked_idx)
        mask_tokens = mask_tokens + pos_masked

        # concatenate visible and masked tokens
        x = torch.cat([x, mask_tokens], dim=1)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # keep masked token predictions
        x = x[:, -masked_idx.shape[1]:]

        # project to encoder embedding dimension and return
        return self.proj(x) # (B, T_masked, embed_dim)
