import collections.abc

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = mge.tensor(1 - drop_prob, dtype=x.dtype)
    # work with diff dim tensors, not just 2D ConvNets
    size = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + mge.random.normal(mean=0, std=1, size=size)
    random_tensor = F.floor(random_tensor)  # binarize
    print(random_tensor)
    output = x / keep_prob * random_tensor
    return output


def to_2tuple(n):
    if isinstance(n, collections.abc.Iterable):
        return x
    return (n, n)


class DropPath(M.Module):

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
