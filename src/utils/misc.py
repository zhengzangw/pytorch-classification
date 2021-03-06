import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only


# https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/0c67dce524b2eb94dc3587ff2832e28f11440cae/utils/utils.py#L26
def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def create_plugins(cfgs):
    plugins = []
    if "SLURM_JOB_ID" in os.environ:
        plugins.append(pl.plugins.environments.SLURMEnvironment())

    if cfgs.trainer.get("accelerator", None) == "ddp":
        plugins.append(
            pl.plugins.DDPPlugin(
                num_nodes=cfgs.trainer.get("num_nodes", 1),
                sync_batchnorm=cfgs.trainer.get("sync_batchnorm", False),
                find_unused_parameters=cfgs.get("find_unused_parameters", False),
            )
        )
    return plugins


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(y.size(0)).type_as(y)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
