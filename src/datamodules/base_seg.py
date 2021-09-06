import glob
import os
import random

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils import data


class Cityscapes_Compatible_Dataset(data.Dataset):
    # mean & std from cityscapes
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        rootpath=None,
        split=None,
        n_classes=19,
        class_info=None,
        augmentations=None,
        ignore_index=250,
        norm=True,
        is_transform=True,
        img_cols=None,
        img_rows=None,
        is_imageio=False,
        superpixel=False,
        **kwargs
    ):
        super().__init__()
        self.root = rootpath
        self.split = split
        self.norm = norm
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.is_transform = is_transform
        self.is_imageio = is_imageio
        self.superpixel = superpixel

        self.valid_classes = class_info[n_classes]["valid"]
        self.class_names = class_info[n_classes]["names"]
        assert self.n_classes == len(self.valid_classes)
        self.class_map = dict(zip(self.valid_classes, range(n_classes)))

        if "colors" in class_info[n_classes]:
            self.colors = class_info[n_classes]["colors"]
            self.label_colours = dict(zip(range(n_classes), self.colors))

        self.augmentations = augmentations
        self.img_size = (img_cols, img_rows)

        if self.is_imageio:
            imageio.plugins.freeimage.download()

    def post_process(self, img_path, lbl_path, sp_path=None):
        img = Image.open(img_path)
        if self.is_imageio:
            lbl = np.asarray(imageio.imread(lbl_path, format="PNG-FI"))[:, :, 0]
            lbl = Image.fromarray(lbl)
        else:
            lbl = Image.open(lbl_path)

        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if sp_path is not None:
            sps = np.load(sp_path, allow_pickle=True)
            sps = Image.fromarray(sps)
            sps = sps.resize(self.img_size, Image.NEAREST)
            sps = np.array(sps, dtype=np.int32)

        if self.augmentations is not None:
            if sp_path is not None:
                img, masks = self.augmentations(img, [lbl, sps])
                lbl, sp = masks
            else:
                img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            if sp_path is not None:
                img, lbl, sp = self.transform(img, lbl, sp=sp)
            else:
                img, lbl = self.transform(img, lbl)

        ret = dict(image=img, label=lbl)
        if sp_path is not None:
            ret["superpixel"] = sp

        return ret

    def transform(self, img, lbl, sp=None):
        lbl = torch.from_numpy(lbl).long()
        img = F.to_tensor(img)
        if self.norm:
            img = F.normalize(img, self.mean, self.std)

        if sp is not None:
            sp = torch.from_numpy(sp).long()
            return img, lbl, sp
        else:
            return img, lbl

    def decode_segmap(self, temp):
        rgb = np.tile(temp, (3, 1, 1)).astype(float).transpose(1, 2, 0)

        for lab in range(0, self.n_classes):
            for i in range(0, rgb.shape[-1]):
                rgb[:, :, i][temp == lab] = self.label_colours[lab][i]

        rgb /= 255.0
        return rgb

    def encode_segmap(self, mask):
        label_copy = self.ignore_index * np.ones(mask.shape, dtype=np.uint8)
        for k, v in list(self.class_map.items()):
            label_copy[mask == k] = v
        return label_copy
