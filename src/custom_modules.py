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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(mlp_ratio * embed_dim))

    def forward(self, x): # (B, T, embed_dim)
        x = x + self.mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x # (B, T, embed_dim)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        assert embed_dim % heads == 0

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads
        self.heads = heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim) # create query, key, and value vector for each token
        self.out_proj = nn.Linear(embed_dim, embed_dim)     # mix information of head outputs

    def forward(self, x):                                                   # (B, T, embed_dim)
        B, T, D = x.shape
        qkv = self.qkv_proj(x)                                              # (B, T, 3*embed_dim)
        qkv = qkv.view(B, T, 3, self.heads, self.head_dim).transpose(1, 3)  # (B, heads, T, 3, head_dim)
        q, k, v = qkv.unbind(dim=3)                                         # (B, heads, T, head_dim)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)     # (B, heads, T, head_dim)
        attn = attn.transpose(1, 2).reshape(B, T, D)                        # (B, T, embed_dim)
        return self.out_proj(attn)                                          # (B, T, embed_dim)
    
class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
    
    def forward(self, x):               # (B, T, dim)
        x = self.act(self.linear1(x))   # (B, T, hidden_dim)
        x = self.linear2(x)             # (B, T, dim)
        return x