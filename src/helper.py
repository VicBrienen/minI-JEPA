import torch

def get_1d_pos_embed(embed_dim, positions):
    # create frequency for every 2 embedding dimensions
    assert embed_dim % 2 == 0
    freqs = torch.arange(embed_dim // 2, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (freqs / (embed_dim // 2)))

    # get phase values
    out = positions[:, None] * freqs[None, :]

    # apply sin and cos to form positional embedding
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)