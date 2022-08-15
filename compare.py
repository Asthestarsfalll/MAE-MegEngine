import time

import megengine as mge
import numpy as np
import torch

from models.mae import mae_vit_base_patch16, mae_vit_huge_patch14, mae_vit_large_patch16
from models.torch_mae import mae_vit_base_patch16 as torch_mae_vit_base_patch16
from models.torch_mae import mae_vit_huge_patch14 as torch_mae_vit_huge_patch14
from models.torch_mae import mae_vit_large_patch16 as torch_mae_vit_large_patch16
from convert_weights import convert

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


mge_model = mae_vit_base_patch16(True)
# mge_model = mae_vit_large_patch16()
# mge_model = mae_vit_huge_patch14()

torch_model = torch_mae_vit_base_patch16()
# torch_model = torch_mae_vit_large_patch16()
# torch_model = torch_mae_vit_huge_patch14()

s = torch_model.state_dict()
m = convert(torch_model, s)
mge_model.load_state_dict(m)

mge_model.eval()
torch_model.eval()

torch_time = meg_time = 0.0

def test_func(mge_out, torch_out):
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def post_process(logits):
    img = (logits * imagenet_std + imagenet_mean)*255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


for i in range(15):
    results = []
    inp = np.random.randn(2, 3, 224, 224)
    mge_inp = mge.tensor(inp, dtype=np.float32)
    torch_inp = torch.tensor(inp, dtype=torch.float32)

    if torch.cuda.is_available():
        torch_inp = torch_inp.cuda()
        torch_model.cuda()

    st = time.time()
    mge_out = mge_model(mge_inp)[0]
    meg_time += time.time() - st

    st = time.time()
    torch_out = torch_model(torch_inp)[0]
    torch_time += time.time() - st

    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    mge_out = post_process(mge_out)
    torch_out = post_process(torch_out)
    ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out)
    results.append(allclose)
    print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")
    assert all(results), "not aligned, make sure you are using gpu device, because megengine gather api don't support large input on cpu"

print(f"meg time: {meg_time}, torch time: {torch_time}")