from functools import partial

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine import hub
from megengine.module import LayerNorm

from .pos_embed import get_2d_sincos_pos_embed
from .utils import DropPath
from .vision_transformer import Block, PatchEmbed


class MaskedAutoencoderViT(M.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=LayerNorm,
        norm_pix_loss=False
    ):
        super(MaskedAutoencoderViT, self).__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mge.Parameter(F.zeros([1, 1, embed_dim]))
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, int(num_patches**.5), cls_token=True)
        self.pos_embed = mge.Parameter(np.expand_dims(
            pos_embed, axis=0), dtype=np.float32)  # fixed sin-cos embedding

        self.blocks = [
            Block(embed_dim,
                  num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  norm_layer=norm_layer)
            for i in range(depth)
        ]

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = M.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = mge.Parameter(F.zeros([1, 1, decoder_embed_dim]))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            decoder_embed_dim, int(num_patches**.5), cls_token=True)
        self.decoder_pos_embed = mge.Parameter(np.expand_dims(
            decoder_pos_embed, axis=0), dtype=np.float32)  # fixed sin-cos embedding

        self.decoder_blocks = [
            Block(decoder_embed_dim,
                  decoder_num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  norm_layer=norm_layer)
            for i in range(decoder_depth)
        ]

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = M.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self._initialize_weights()

    def _initialize_weights(self):

        # initialize patch_embed like M.Linear (instead of M.Conv2d)
        w = self.patch_embed.proj.weight
        M.init.xavier_uniform_(w.reshape([w.shape[0], -1]))

        # trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        M.init.normal_(self.cls_token, std=.02)
        M.init.normal_(self.mask_token, std=.02)

        # initialize M.Linear and M.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, M.Linear):
            # use xavier_uniform following official JAX ViT:
            M.init.xavier_uniform_(m.weight)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.zeros_(m.bias)
        elif isinstance(m, LayerNorm):
            M.init.zeros_(m.bias)
            M.init.ones_(m.weight)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        N, _, ori_h, ori_w = imgs.shape
        assert ori_h == ori_w, "The height and width of input image must be equal"
        assert ori_h % p == 0, "The input image size must be a multiple of patch size"

        window_size = imgs.shape[2] // p
        x = imgs.reshape((N, 3, window_size, p, window_size, p))
        x = x.transpose((0, 2, 4, 3, 5, 1))
        x = x.reshape((N, window_size**2, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        N, L = x.shape[:2]
        ori_h = ori_w = int(L**.5)
        assert ori_h * ori_w == x.shape[1]

        x = x.reshape((N, ori_h, ori_w, p, p, 3))
        x = x.transpose((0, 5, 1, 3, 2, 4))
        imgs = x.reshape((N, 3, ori_h * p, ori_w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        N, L, C = x.shape
        len_keep = int(L * (1 - mask_ratio))
        np.random.seed(22)
        noise = mge.tensor(np.random.rand(N, L))
        # noise = mge.random.normal(size=[N, L])

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = F.argsort(noise)
        ids_restore = F.argsort(ids_shuffle)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        o_ids_keep = ids_keep
        ids_keep = F.expand_dims(ids_keep, axis=-1)
        x_masked = F.gather(x, axis=1, index=F.repeat(ids_keep, C, axis=2))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = F.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = F.gather(mask, axis=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = F.repeat(cls_token, repeats=x.shape[0], axis=0)
        x = F.concat([cls_tokens, x], axis=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        B, L, C = x.shape
        # append mask tokens to sequence

        mask_tokens = F.broadcast_to(self.mask_token, shape=[
                                     B, ids_restore.shape[1] + 1 - x.shape[1], self.mask_token.shape[-1]])
        x_ = F.concat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        ids_restore = F.expand_dims(ids_restore, axis=-1)
        x_ = F.gather(x_, axis=1, index=F.repeat(ids_restore, C, axis=2))
        x = F.concat([x[:, :1, :], x_], axis=1)  # append cls token

        # add pos embed

        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = F.mean(target, axis=-1, keepdims=True)
            var = F.var(target, axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = F.mean(loss, axis=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/34/files/c3a51b70-1d86-485d-a31b-078a5d2eac26"
)
def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model

