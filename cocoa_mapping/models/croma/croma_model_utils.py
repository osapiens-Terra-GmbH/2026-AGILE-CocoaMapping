"""The code is based on the code from https://github.com/antofuller/CROMA/tree/main under MIT license.

Citation:
    @inproceedings{fuller2023croma,
        title={CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders},
        author={Fuller, Anthony and Millard, Koreen and Green, James R},
        booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
        year={2023}}
"""

from contextlib import contextmanager
import itertools
import math

import numpy as np
import torch


def get_alibi(num_heads, num_patches):
    # inspired by: https://github.com/ofirpress/attention_with_linear_biases
    side_length = int(math.sqrt(num_patches))
    points = list(itertools.product(range(side_length), range(side_length)))
    points = torch.tensor(points, dtype=torch.float32)

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
            :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(
        1).unsqueeze(2).unsqueeze(3)

    # Use broadcasting instead of looping
    p1 = points.unsqueeze(0)
    p2 = points.unsqueeze(1)
    dist = torch.cdist(p1, p2)

    all_bias = dist.unsqueeze(0).unsqueeze(0) * slopes * -1
    return all_bias.view(1, num_heads, num_patches, num_patches)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


@contextmanager
def eval_mode(model: torch.nn.Module):
    """
    Context manager for setting a PyTorch model to evaluation mode and restoring
    its previous training mode on exit.

    Args:
        model (torch.nn.Module): The PyTorch model to manage.
    """
    # Save the current mode (True if training, False if in eval)
    current_mode = model.training

    # Switch to eval mode
    model.eval()

    # Set torch.no_grad
    try:
        with torch.no_grad():
            yield
    finally:
        # Restore the original mode
        model.train(current_mode)
