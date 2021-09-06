from typing import Any, List, Optional

import einops
import hydra
import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from ..modules.memqueue import MemQueue
from ..optimizer.scheduler import create_scheduler
from ..utils import utils
from ..utils.misc import mixup_data
from ..utils.tools import set_bn_momentum, strip_ignore_index
from .base import LitBase

log = utils.get_logger(__name__)


def downsample_mask(labels, size=(256, 512)):
    labels_ = einops.rearrange(labels, "b h w -> b 1 h w")
    labels_ = F.interpolate(labels_.float(), size=size, mode="nearest").long().flatten()
    return labels_


def list_np_to_torch(np_list):
    ret = []
    for np_array in np_list:
        ret.append(torch.from_numpy(np_array).cuda())
    return ret


class LitSegDA(LitBase):
    def __init__(self, cfg: Optional[DictConfig]):
        super().__init__(cfg=cfg)

        # metric
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(),
                torchmetrics.IoU(self.config.datamodule.num_classes).double(),
            ]
        )
        self.train_src_metrics = metrics.clone(prefix="train/src_")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # ssl
        self._init_memqueue()
        # breakpoint()

    # ------------
    # ssl
    # ------------

    def _init_memqueue(self):
        self.ssl = self.config.ssl.enable
        k = list(self.config.ssl.k)
        k = k if isinstance(k, list) else [k]
        k_rep = self.config.ssl.k_rep
        self.k_list = []
        for i in k:
            self.k_list += k_rep * [i]

        self.memqueue_tgt = self.memqueue_src = None
        if self.ssl:
            self.memqueue_src = MemQueue()
            self.proto_src = None
            if self.config.datamodule.src != self.config.datamodule.tgt:
                self.memqueue_tgt = MemQueue()
                self.proto_tgt = None

    def cal_feature_by_superpixel(self, features, labels, superpixels):
        resolution = features.shape[-2:]

        # prepare features
        features = features.detach()
        features = einops.rearrange(features, "b c h w -> (b h w) c")
        features = F.normalize(features, dim=1, p=2, eps=1e-12)

        # resize label and superpixel
        labels = downsample_mask(labels, size=resolution)
        superpixels = downsample_mask(superpixels, size=resolution)
        mask = labels != self.config.datamodule.ignore_index

        labels = labels[mask]
        superpixels = superpixels[mask]
        features = features[mask]

        max_superpixel_id = 22000
        mean_features = torch.zeros((max_superpixel_id, 256), device=self.device)

        mean_features.scatter_add_(0, superpixels.unsqueeze(1).expand((-1, 256)), features)

        zero_mask = mean_features.norm(dim=-1) < 1e-12
        mean_features = mean_features[~zero_mask]
        mean_features = mean_features.detach().cpu().numpy()

        return mean_features

    def cal_proto(self):
        pass

    # ------------
    # train
    # ------------

    def training_step(self, batch: Any, batch_idx: int):
        # Get src info
        batch_src = batch["src"]
        src_imgs, src_labels, src_idxs = (
            batch_src["image"],
            batch_src["label"],
            batch_src["index"],
        )

        # Get tgt info
        if "tgt" in batch:
            batch_tgt = batch["tgt"]
            tgt_imgs, tgt_labels, tgt_idxs = (
                batch_tgt["image"],
                batch_tgt["label"],
                batch_tgt["index"],
            )

        # loss
        src_results = self.forward(src_imgs)
        src_logits = src_results["out"]
        src_loss = self.criterion(src_logits, src_labels)

        if self.ssl:
            tgt_results = self.forward(tgt_imgs)

        loss = src_loss
        ret = dict(loss=loss)

        # ssl update
        if self.memqueue_src:
            src_mean_features = self.cal_feature_by_superpixel(
                src_results["feature"], src_labels, batch_src["superpixel"]
            )
            self.memqueue_src.push(src_mean_features)
        if self.memqueue_tgt:
            tgt_mean_features = self.cal_feature_by_superpixel(
                tgt_results["feature"], tgt_labels, batch_tgt["superpixel"]
            )
            self.memqueue_tgt.push(tgt_mean_features)

        # metric
        preds = src_logits.argmax(dim=1)
        preds, labels = strip_ignore_index(preds, src_labels, self.config.datamodule.ignore_index)
        self.train_src_metrics(preds, labels)
        self.log("train/loss", loss)

        return ret

    def training_step_end(self, outputs):
        # ssl
        if self.memqueue_src.filled:
            proto_src_np = self.memqueue_src.protos(self.k_list)
            self.proto_src = list_np_to_torch(proto_src_np)
        if self.memqueue_tgt.filled:
            proto_tgt_np = self.memqueue_tgt.protos(self.k_list)
            self.proto_tgt = list_np_to_torch(proto_tgt_np)

        return outputs

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_src_metrics.compute(), prog_bar=True)
        self.train_src_metrics.reset()

    # ------------
    # validation
    # ------------

    def validation_step(self, batch: Any, batch_idx: int):
        imgs, labels, idxs = batch["image"], batch["label"], batch["index"]

        # loss
        result = self.model(imgs)
        logits = result["out"]
        loss = self.criterion(logits, labels)

        # metric
        pred = logits.argmax(dim=1)
        pred, labels = strip_ignore_index(pred, labels, self.config.datamodule.ignore_index)
        self.valid_metrics(pred, labels)
        self.log("val/loss", loss)

        return loss

    def validation_epoch_end(self, outputs):
        self.log_dict(self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    # ------------
    # test
    # ------------

    def test_step(self, batch: Any, batch_idx: int):
        imgs, labels, idxs = batch

        # loss
        result = self.model(imgs)
        logits = result["out"]
        loss = self.criterion(logits, labels)

        # metric
        pred = logits.argmax(dim=1)
        pred, labels = strip_ignore_index(pred, labels, self.config.datamodule.ignore_index)
        self.test_metrics(pred, labels)
        self.log("test/loss", loss)

        return loss

    def test_epoch_end(self, outputs):
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):

        # === optimizer ===
        set_bn_momentum(self.model.backbone, momentum=0.01)
        lr = self.config.optimizer.lr
        optimizer = torch.optim.SGD(
            params=[
                {
                    "params": self.model.backbone.parameters(),
                    "lr": lr,
                },
                {"params": self.model.aspp.parameters(), "lr": 10 * lr},
                {
                    "params": self.model.decoder.parameters(),
                    "lr": 10 * lr,
                },
            ],
            lr=lr,
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay,
        )

        # === Scheduler ===
        schedulers = self.configure_scheduler(optimizer)

        return [optimizer], schedulers
