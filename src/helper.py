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

def get_2d_pos_embed(embed_dim, grid_size):
    # evenly between height and width
    assert embed_dim % 2 == 0
    grid_coords = torch.arange(grid_size, dtype=torch.float32)

    # create 2d grid
    grid_h, grid_w = torch.meshgrid(grid_coords, grid_coords, indexing="ij")
    grid = torch.stack([grid_h, grid_w]).reshape(2, -1)

    # encode height and width seperately
    emb_h = get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_pos_embed(embed_dim // 2, grid[1])

    # concatenate height and width to form final 2d embedding
    return torch.cat([emb_h, emb_w], dim=1)

def gather_tokens(x, idx):
    idx = idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
    return torch.gather(x, dim=1, index=idx)

@torch.no_grad()
def update_ema(student, teacher, decay):
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(decay).add_(p_s.data * (1.0 - decay))