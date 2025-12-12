import torch
from torch import nn
from torch.nn import functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # embed using convolution with stride = patch_size
        self.embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
    
    def forward(self, x):           # (B, C, H, W)
        x = self.embed(x)           # (B, D, H, W)
        x = x.flatten(2)            # (B, D, T)
        return x.transpose(1, 2)    # (B, T, D)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        assert embed_dim % heads == 0

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads
        self.heads = heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim) # create query, key, and value vector for each token
        self.out_proj = nn.Linear(embed_dim, embed_dim)     # mix information of head outputs

    def forward(self, x):                                                   # (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv_proj(x)                                              # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.heads, self.head_dim).transpose(1, 3)  # (B, H, T, 3, HD)
        q, k, v = qkv.unbind(dim=3)                                         # (B, H, T, HD)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)     # (B, H, T, HD)
        attn = attn.transpose(1, 2).reshape(B, T, D)                        # (B, T, D)
        return self.out_proj(attn)                                          # (B, T, D)