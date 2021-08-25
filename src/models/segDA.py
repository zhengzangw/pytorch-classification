from typing import Any, List, Optional

import hydra
import torch
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from ..optimizer.scheduler import create_scheduler
from ..utils import utils
from ..utils.misc import mixup_data
from ..utils.tools import set_bn_momentum, strip_ignore_index
from .base import LitBase

log = utils.get_logger(__name__)


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

    # ------------
    # train
    # ------------

    def training_step(self, batch: Any, batch_idx: int):
        src_imgs, src_labels, src_idxs = batch["src"]
        if "tgt" in batch:
            tgt_imgs, tgt_labels, tgt_idxs = batch["tgt"]

        # loss
        src_results = self.forward(src_imgs)
        src_logits = src_results["out"]
        src_loss = self.criterion(src_logits, src_labels)

        loss = src_loss

        # metric
        preds = src_logits.argmax(dim=1)
        preds, labels = strip_ignore_index(preds, src_labels, self.config.datamodule.ignore_index)
        self.train_src_metrics(preds, labels)

        self.log("train/loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_src_metrics.compute(), prog_bar=True)
        self.train_src_metrics.reset()

    # ------------
    # validation
    # ------------

    def validation_step(self, batch: Any, batch_idx: int):
        imgs, labels, idxs = batch

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

    def test_step(self, batch: Any):
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
