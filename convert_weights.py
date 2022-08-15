import argparse
import os

import megengine as mge
import numpy as np
import torch
import torch.nn as nn

from models.mae import mae_vit_base_patch16, mae_vit_huge_patch14, mae_vit_large_patch16
from models.torch_mae import mae_vit_base_patch16 as torch_mae_vit_base_patch16
from models.torch_mae import mae_vit_large_patch16 as torch_mae_vit_large_patch16
from models.torch_mae import mae_vit_huge_patch14 as torch_mae_vit_huge_patch14


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


def convert(torch_model, torch_dict):
    new_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_model, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        new_dict[k] = data
    return new_dict


def main(torch_name, torch_path):
    # download manually if speed is too slow
    torch_state_dict = torch.load(torch_path, map_location='cpu')['model']
    torch_model = eval("torch_" + torch_name)()
    # torch_state_dict = torch_model.state_dict()
    torch_model.load_state_dict(torch_state_dict)
    # torch.save(torch_state_dict, './base.pth')
    new_dict = convert(torch_model, torch_state_dict)
    model = eval(torch_name)()

    model.load_state_dict(new_dict)
    os.makedirs('pretrained', exist_ok=True)
    mge.save(new_dict, os.path.join('pretrained', torch_name + '.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='mae_vit_base_patch16',
        help="which model to convert from torch to megengine, default: mae_vit_base_patch16, optional: mae_vit_base_patch16, torch_mae_vit_large_patch16, torch_mae_vit_huge_patch14",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        default=None,
        help=f"Path to torch saved model, default: None",
    )
    args = parser.parse_args()
    main(args.model, args.ckpt)
