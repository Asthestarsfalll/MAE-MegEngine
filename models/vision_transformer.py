import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine.module import GELU, LayerNorm

from .utils import DropPath, to_2tuple


class PatchEmbed(M.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True
    ):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = M.Conv2d(in_chans, embed_dim,
                             kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else M.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f'Input image size does not match, expected {self.img_size}, got {(H, W)}'
        x = self.proj(x)
        if self.flatten:
            x = F.flatten(x, 2).transpose(0, 2, 1)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(M.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0. ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = M.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = M.Dropout(attn_drop)
        self.proj = M.Linear(dim, dim)
        self.proj_drop = M.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # B, N, 3, H, C // H
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # 3, B, H, N, C // H
        qkv = qkv.transpose((2, 0, 3, 1, 4))
        # B, H, N, C // H
        split_out = F.split(qkv, 3)
        split_out = [F.squeeze(i, 0) for i in split_out]
        q, k, v = split_out

        attn = (q @ k.transpose((0, 1, 3, 2))) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose((0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(M.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = mge.Parameter(init_values * F.ones([dim]))

    def forward(self, x):
        return x * self.gamma


class Mlp(M.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = M.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = M.Linear(hidden_features, out_features)
        self.drop = M.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(M.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        drop=0., 
        attn_drop=0., 
        init_values=None,
        drop_path=0., 
        act_layer=GELU, 
        norm_layer=LayerNorm
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else M.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else M.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else M.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else M.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
