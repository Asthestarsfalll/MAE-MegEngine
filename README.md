# MAE-MegEngine

The MegEngine Implementation of MAE(Masked Auto Encoder).

## Usage

**Make sure  you are using a GPU device, for there is a gap between output of GPU and CPU device in MegEngine gather API** 

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't want to compare the ouput error between the MegEngine implementation and PyTorch one, just ignore requirements.txt and install MegEngine from the command line:

```python
python3 -m pip install --upgrade pip 
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

 Note:

The pytorch implementation is based on timm==`0.3.2`, for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### Convert Weights

Convert trained weights from torch to megengine, the converted weights will be save in ./pretained/ , you need to specify the converte model architecture and path to checkpoint offered by official repo.

pre-trained checkpoint:

| ViT-Base                                                     | ViT-Large                                                    | ViT-Huge                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) | [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth) |

visuialize checkpoint:

| ViT-Base                                                     | ViT-Large                                                    | ViT-Large-GanLoss                                            | ViT-Huge                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [download](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth) | [download](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth) | [download](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth) | [download](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_huge.pth) |

```bash
python convert_weights.py -m mae_vit_base_patch16 -c /local/path/to/ckpt
```

### Compare

Use `python compare.py` .

By default, the compare script will convert the torch state_dict to the format that megengine need.

If you want to compare the error by checkpoints, you neet load them manually.

### Visualize

Just read  and run `visualize.py`.

### Load From Hub

Import from megengine.hub:

Way 1:

```python
from functools import partial
import megengine.module as M
from megengine import hub

modelhub = hub.import_module(
    repo_info='asthestarsfalll/MAE-MegEngine:main', git_host='github.com')

# load VAN model and custom on you own
model = modelhub.MAE(
    patch_size=16, embed_dim=768, depth=12, num_heads=12,
    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    mlp_ratio=4, norm_layer=partial(M.LayerNorm, eps=1e-6))

# load pretrained model
pretrained_model = modelhub.mae_vit_base_patch16(pretrained=True)

```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'mae_vit_base_patch16'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/MAE-MegEngine:main', entry=model_name, git_host='github.com', pretrained=True)
```

*Currently pretrained model only support mae_vit_base_patch16.*

But you can still load the model without pretrained weights like this:

```python
model = modelhub.mae_vit_large_patch16()
# or
model_name = 'mae_vit_large_patch16'
model = hub.load(
    repo_info='asthestarsfalll/MAE-MegEngine:main', entry=model_name, git_host='github.com')
```

## TODO

- [ ] Add interfaces of visialize.
- [ ] Some down stream tasks maybe.
- [ ] Some introduction about MAE.

## Reference

[The official implementation of MAE](https://github.com/facebookresearch/mae/)
