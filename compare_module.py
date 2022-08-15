import time
from functools import partial

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF

from models.mae import MaskedAutoencoderViT as MAE
from models.torch_mae import Block as TorchBlock
from models.torch_mae import MaskedAutoencoderViT as TorchMAE
from models.vision_transformer import Attention, PatchEmbed, Mlp, Block
from timm.models.vision_transformer import Attention as timm_Attention
from timm.models.vision_transformer import PatchEmbed as timm_PatchEmbed
from timm.models.vision_transformer import Mlp as timm_Mlp


GLOBAL_RTOL = 1e-3
BATCH_SIZE = 8

DTYPE_MAPPER = {
    # 'float16': (np.float16, torch.float16),
    'float32': (np.float32, torch.float32),
    # 'float64': (np.float64, torch.float64),
}

INPUT_SHAPE = {
    "Attention": [
        (BATCH_SIZE, 32, 512),
    ],
    "MLP": [
        (BATCH_SIZE, 32, 768),
    ],
    "PatchEmbed": [
        (BATCH_SIZE, 3, 224, 224),
    ],
    "Block": [
        (BATCH_SIZE, 64, 768),
    ],
    "MAE": [
        (BATCH_SIZE, 3, 224, 224)
        # (BATCH_SIZE, 5, 10),
    ],
    "Random Masking": [
        (BATCH_SIZE, 196, 768)
    ],
    "Encoder": [
        (BATCH_SIZE, 3, 224, 224)
    ],
}


KWARDS_MAPPER = {
    "Attention": [
        dict(dim=512, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
    ],
    "MLP": [
        dict(in_features=768, hidden_features=3072, out_features=None, drop=0.)
    ],
    "PatchEmbed": [
        dict(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    ],
    "Block": [
        dict(dim=768, num_heads=8, mlp_ratio=4., qkv_bias=False,
             drop=0., attn_drop=0., drop_path=0.)
    ],
    "MAE": [
        dict(img_size=224, patch_size=16, in_chans=3, embed_dim=32, depth=2, num_heads=2,
             decoder_embed_dim=32, decoder_depth=4, decoder_num_heads=2, mlp_ratio=2.)
    ],

}


class FunctionWrapper():
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x)

    def eval(self):
        return
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, x):
        return

def generate_inputs(shape, dtype='float32'):
    inp = np.random.randn(*shape)
    types = DTYPE_MAPPER[dtype]
    mge_inp = mge.tensor(inp, dtype=types[0])
    torch_inp = torch.tensor(inp, dtype=types[1])
    return mge_inp, torch_inp


def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module


def convert_state_dict(torch_module, torch_dict):
    mge_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_module, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        mge_dict[k] = data

    return mge_dict

M = MAE(**KWARDS_MAPPER["MAE"][0])
T = TorchMAE(**KWARDS_MAPPER["MAE"][0])
s = T.state_dict()
m = convert_state_dict(T, s)
M.load_state_dict(m)
M.eval()
T.eval()

CLASS_MAPPER = {
    # "Attention": (Attention, timm_Attention),
    # "MLP": (Mlp, timm_Mlp),
    # "PatchEmbed": (PatchEmbed, timm_PatchEmbed),
    # "Block": (Block, TorchBlock),
    # "MAE": (MAE, TorchMAE),
    # "Random Masking": (
    #     partial(
    #         FunctionWrapper,
    #         func=partial(
    #             M.random_masking,
    #             mask_ratio=0.75
    #         )
    #     ),
    #     partial(
    #         FunctionWrapper,
    #         func=partial(
    #             T.random_masking,
    #             mask_ratio=0.75
    #         )
    #     )
    # ),
    "Encoder": (
        partial(
            FunctionWrapper,
            func=partial(
                M.forward_encoder,
                mask_ratio=0.75
            )
        ),
        partial(
            FunctionWrapper,
            func=partial(
                T.forward_encoder,
                mask_ratio=0.75
            )
        )
    ),
}


def is_in_string(targets: list, s: str):
    return any(t in s for t in targets)


def convert_dtype(m):
    pass


def test_func(mge_tensor, torch_tensor):
    mge_out = mge_tensor.numpy()
    if torch.cuda.is_available():
        torch_out = torch_tensor.detach().cpu().numpy()
    else:
        torch_out = torch_tensor.detach().numpy()
    result = np.isclose(mge_out, torch_out, rtol=GLOBAL_RTOL)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def get_channels(kwards):
    for n in ['inplanes', 'in_channels']:
        if n in kwards:
            ch = kwards[n]
            if isinstance(ch, list):
                return ch
            return [ch]
    else:
        # if 'dims' in kwards:
        #     return [3]
        return list(np.random.randint(1, 2048, size=[1]))


def main():
    print(f"Begin test with rtol = {GLOBAL_RTOL}, batch size ={BATCH_SIZE}")
    print()
    unalign_list = []
    a = 0
    for k, (mge_class, torch_class) in CLASS_MAPPER.items():
        kwards = KWARDS_MAPPER.get(k, [{}])
        print(f"Begin test {k}:")
        for kw in kwards:
            print(f"\t with kwards {kw}:")
            torch_module = torch_class(**kw)
            torch_module.eval()
            mge_module = mge_class(**kw)
            mge_module.eval()
            for input_shape in INPUT_SHAPE[k]:
                for dtype in DTYPE_MAPPER.keys():
                    mge_inp, torch_inp = generate_inputs(input_shape, dtype)
                    print(f"\t\t with shape {mge_inp.shape}:")
                    print(f"\t\t\t with dtype {dtype}:")
                    torch_dict = torch_module.state_dict()
                    mge_dict = convert_state_dict(torch_module, torch_dict)
                    mge_module.load_state_dict(mge_dict)

                    st = time.time()
                    mge_out = mge_module(mge_inp)
                    # mge_out = mge_module.random_masking(mge_inp, mask_ratio=0.75)
                    mge_time = time.time() - st

                    st = time.time()
                    torch_out = torch_module(torch_inp)
                    # torch_out = torch_module.random_masking(torch_inp, mask_ratio=0.75)
                    torch_time = time.time() - st
                    global_allcolse = True
                    if isinstance(mge_out, (list, tuple)):
                        for i, (m, t) in enumerate(zip(mge_out, torch_out)):
                            print(f'The {i+1}th element of output:')
                            ratio, allclose, abs_err, std_err = test_func(m, t)
                            if not allclose:
                                global_allcolse = False
                            print(
                                f"\t\t\t\tResult: {allclose}, {ratio*100 : .4f}% elements is close enough\n \t\t\t\t which absolute error is  {abs_err} and absolute std is {std_err}")
                    elif isinstance(mge_out, dict):
                        for k in mge_out.keys():
                            print(f"Key: {k}")
                            m = mge_out[k]
                            t = torch_out[k]
                            ratio, allclose, abs_err, std_err = test_func(m, t)
                            if not allclose:
                                global_allcolse = False
                            print(
                                f"\t\t\t\tResult: {allclose}, {ratio*100 : .4f}% elements is close enough\n \t\t\t\t which absolute error is  {abs_err} and absolute std is {std_err}")
                    else:
                        ratio, global_allcolse, abs_err, std_err = test_func(
                            mge_out, torch_out)
                        print(
                            f"\t\t\t\tResult: {global_allcolse}, {ratio*100 : .4f}% elements is close enough\n \t\t\t\t which absolute error is  {abs_err} and absolute std is {std_err}")
                    if not global_allcolse:
                        unalign_list.append(k)
                    print(
                        f"\t\t\t\ttime used: megengine: {mge_time : .4f}s, torch: {torch_time : .4f}s")
    print(f"Test down, unaligned module: {list(set(unalign_list))}")


if __name__ == "__main__":
    main()
