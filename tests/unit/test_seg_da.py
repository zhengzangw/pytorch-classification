import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src import datamodules
from src.augmentations.seg_aug import get_composed_augmentations
from src.datamodules.cityscapes import Cityscapes
from src.datamodules.datamodule_seg import DADataModule
from src.datamodules.gta5 import GTA5
from src.datamodules.synthia import Synthia

aug = {
    "rcrop": [768, 768],
    "brightness": 0.5,
    "contrast": 0.5,
    "saturation": 0.5,
    "hflip": 0.5,
}


def template_test(data_class, name, **kwargs):
    augmentations = get_composed_augmentations(aug)

    dst = data_class(augmentations=augmentations, **kwargs)
    bs = 4
    trainloader = torch.utils.data.DataLoader(dst, batch_size=bs, num_workers=0)

    batch = next(iter(trainloader))
    imgs, labels, ind = batch["image"], batch["label"], batch["index"]
    imgs = imgs.numpy()[:, ::-1, :, :]
    imgs = np.transpose(imgs, [0, 2, 3, 1])
    f, axarr = plt.subplots(bs, 2)
    for j in range(bs):
        axarr[j][0].set_axis_off()
        axarr[j][1].set_axis_off()
        axarr[j][0].imshow(imgs[j])
        axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
    plt.savefig(f"{name}_test.png")


def test_Cityscapes():
    return template_test(
        Cityscapes,
        "cityscapes",
        rootpath="./data/cityscapes",
        split="train",
        n_classes=19,
        img_cols=2048,
        img_rows=1024,
        norm=False,
    )


def test_GTA5():
    return template_test(
        GTA5,
        "gta5",
        rootpath="./data/GTA5",
        n_classes=19,
        img_cols=1914,
        img_rows=1052,
        norm=False,
    )


def test_datamodule_seg():
    loader = DADataModule(augmentations=aug)
    loader.setup()
    assert not loader.is_da_task

    it = loader.train_dataloader()["src"]
    ret = next(iter(it))
    _, _, _ = ret

    it = loader.val_dataloader()
    ret = next(iter(it))
    _, _, _ = ret
