import os
import sys

import cv2
import matplotlib.pyplot as plt
import megengine as mge
import megengine.functional as F
import numpy as np
import requests
from PIL import Image

from models import mae

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def show_image(image, title=''):
    print("show image")
    # image is [H, W, 3]
    assert image.shape[2] == 3
    image = image.astype(np.float32)
    image = (image * imagenet_std + imagenet_mean) * 255
    img = np.clip(image, 0, 255).astype(np.uint8)
    # cv2.imshow(title, img)
    # cv2.waitKey(0)
    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(mae, arch)()
    # load model
    checkpoint = mge.load(chkpt_dir)
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model


def run_one_image(img, model):
    x = mge.tensor(img, dtype=np.float32)

    print("make it a batch-like")
    x = F.expand_dims(x, 0)
    x = x.transpose((0, 3, 1, 2))

    print("run MAE")
    loss, y, mask = model(x, mask_ratio=0.75)
    y = model.unpatchify(y)
    y = y.transpose((0, 2, 3, 1))

    print("visualize the mask")
    # (N, H*W, p*p*3)
    mask = F.expand_dims(mask, -1)
    mask = F.repeat(mask, model.patch_embed.patch_size[0]**2 * 3, 2)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = mask.transpose((0, 2, 3, 1)) 

    x = x.transpose((0, 2, 3, 1))

    print("masked image")
    im_masked = x * (1 - mask)

    print("MAE reconstruction pasted with visible patches")
    im_paste = x * (1 - mask) + y * mask

    print("make the plt figure larger")
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()


def main():

    # load an image
    # fox, from ILSVRC2012_val_00046145
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'
    # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    plt.rcParams['figure.figsize'] = [5, 5]
    show_image(img)

    chkpt_dir = './pretrained/mae_vit_large_patch16.pkl'
    chkpt_dir = './pretrained/mae_vit_base_patch16.pkl'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
    print('Model loaded.')


    print('MAE with pixel reconstruction:')
    run_one_image(img, model_mae)


if __name__ == "__main__":
    main()
