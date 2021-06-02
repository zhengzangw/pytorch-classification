# https://github.com/lucidrains/mlp-mixer-pytorch
from functools import partial

from einops.layers.torch import Rearrange, Reduce
from torch import nn


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, hidden_dim, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(hidden_dim, dim),
        nn.Dropout(dropout),
    )


def MLPMixer(
    *,
    image_size,
    patch_size,
    dim,
    depth,
    num_classes,
    expansion_factor_last=4,
    expansion_factor_first=0.5,
    dropout=0.0
):
    assert (image_size % patch_size) == 0, "image must be divisible by patch size"
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * 3, dim),
        *[
            nn.Sequential(
                PreNormResidual(
                    dim,
                    FeedForward(
                        num_patches,
                        int(expansion_factor_first * dim),
                        dropout,
                        chan_first,
                    ),
                ),
                PreNormResidual(
                    dim,
                    FeedForward(dim, int(expansion_factor_last * dim), dropout, chan_last),
                ),
            )
            for _ in range(depth)
        ],
        nn.LayerNorm(dim),
        Reduce("b n c -> b c", "mean"),
        nn.Linear(dim, num_classes)
    )


def mlp_mixer_b_16(num_classes=10):
    return MLPMixer(image_size=224, patch_size=16, dim=768, depth=12, num_classes=num_classes)
