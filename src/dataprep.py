import torch

def sample_mask(grid_size, block_size):
    # generate top left indice
    h = torch.randint(0, grid_size - block_size + 1, (1,))
    w = torch.randint(0, grid_size - block_size + 1, (1,))

    # create 2d boolean mask over the patch grid (true = masked, false = visible)
    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    mask[h:h+block_size, w:w+block_size] = True

    mask = mask.flatten()

    # indices of visible and masked tokens
    visible_idx = torch.nonzero(~mask).squeeze(1)
    masked_idx = torch.nonzero(mask).squeeze(1)

    return visible_idx, masked_idx