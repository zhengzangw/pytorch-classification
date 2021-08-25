import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..augmentations.seg_aug import get_composed_augmentations
from .cityscapes import Cityscapes
from .gta5 import GTA5
from .synthia import Synthia

DATA_INFO = {
    "cityscapes": {
        "rootpath": "cityscapes",
        "w": 2048,
        "h": 1024,
        "loader": Cityscapes,
    },
    "cityscapes_resize": {
        "rootpath": "cityscapes",
        "w": 1024,
        "h": 512,
        "loader": Cityscapes,
    },
    "synthia": {
        "rootpath": "SYNTHIA",
        "w": 1914,
        "h": 1052,
        "loader": Synthia,
    },
    "synthia_resize": {
        "rootpath": "SYNTHIA",
        "w": 1280,
        "h": 760,
        "loader": Synthia,
    },
    "gta5": {"rootpath": "GTA5", "w": 1914, "h": 1052, "loader": GTA5},
    "gta5_resize": {
        "rootpath": "GTA5",
        "w": 1280,
        "h": 720,
        "loader": GTA5,
    },
}


def get_cfg(name, n_class=19, data_dir="data/", superpixel=False):
    return {
        "name": name,
        "n_classes": n_class,
        "rootpath": os.path.join(data_dir, DATA_INFO[name]["rootpath"]),
        "img_cols": DATA_INFO[name]["w"],
        "img_rows": DATA_INFO[name]["h"],
    }


class DADataModule(pl.LightningDataModule):
    has_teardown_None = False

    def __init__(
        self,
        data_dir="data/",
        src="cityscapes",
        tgt="cityscapes",
        num_classes=19,
        augmentations=None,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        superpixel=False,
        **kwargs
    ):
        super().__init__()

        self.src = src
        assert src in DATA_INFO
        self.tgt = tgt
        assert tgt.startswith("cityscapes")
        self.is_da_task = src != tgt
        self.n_class = num_classes

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.augmentations = get_composed_augmentations(augmentations)

        self.src_cfg = get_cfg(src, n_class=self.n_class, data_dir=data_dir)
        if self.is_da_task:
            self.tgt_cfg = get_cfg(tgt, n_class=self.n_class, data_dir=data_dir)

        self.superpixel = superpixel
        if superpixel:
            print("Load superpixel.")

    @property
    def train_len(self):
        assert self.src_train_dst is not None
        assert len(self.src_train_dst) > 0
        return len(self.src_train_dst)

    def setup(self, stage=None):
        src_cls = DATA_INFO[self.src]["loader"]
        if self.is_da_task:
            tgt_cls = DATA_INFO[self.tgt]["loader"]

        if stage == "fit" or stage is None:
            # src dataset
            self.src_train_dst = src_cls(
                **self.src_cfg,
                augmentations=self.augmentations,
                split="train",
                superpixel=self.superpixel
            )

            if self.is_da_task:
                # tgt dataset
                self.tgt_train_dst = tgt_cls(
                    **self.tgt_cfg,
                    augmentations=self.augmentations,
                    split="train",
                    superpixel=self.superpixel
                )
                # val dataset
                self.val_dst = tgt_cls(**self.tgt_cfg, augmentations=None, split="train")
            else:
                # val dataset
                self.val_dst = src_cls(**self.src_cfg, augmentations=None, split="val")

        if stage == "test" or stage is None:
            if self.is_da_task:
                self.test_dst = tgt_cls(**self.tgt_cfg, augmentations=None, split="train")
            else:
                self.test_dst = src_cls(**self.src_cfg, augmentations=None, split="val")

    def train_dataloader(self):

        src_loader = DataLoader(
            self.src_train_dst,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )
        ret = dict(src=src_loader)
        if self.is_da_task:
            tgt_loader = DataLoader(
                self.tgt_train_dst,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                drop_last=True,
            )
            ret["tgt"] = tgt_loader

        return ret

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dst,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dst,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_loader
