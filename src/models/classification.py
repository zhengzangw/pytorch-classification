from typing import Any, List, Optional

import hydra
import torch
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from ..optimizer.scheduler import create_scheduler
from ..utils import utils
from ..utils.misc import mixup_data
from .base import LitBase

log = utils.get_logger(__name__)


class LitClassification(LitBase):
    def __init__(self, cfg: Optional[DictConfig], mixup: Optional[float]):
        super().__init__(cfg=cfg, mixup=mixup)

        # metric
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy()])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    # ------------
    # train
    # ------------

    def step(self, batch: Any):
        x, y = batch
        if self.hparams.get("mixup"):
            x, y_a, y_b, lam = mixup_data(x, y, self.hparams.mixup)
            logits = self.forward(x)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
        else:
            logits = self.forward(x)
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        if self.hparams.get("mixup") is None:
            metric = self.train_metrics(preds, targets)
            self.log_dict(metric, prog_bar=True)
        self.log("train/loss", loss)

        return {"loss": loss}

    # ------------
    # validation
    # ------------

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        metric = self.val_metrics(preds, targets)
        self.log_dict(metric, prog_bar=True)
        self.log("val/loss", loss)

        return {"loss": loss, "preds": preds, "targets": targets}

    # ------------
    # test
    # ------------

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        metric = self.val_metrics(preds, targets)
        self.log_dict(metric)
        self.log("test/loss", loss)

        return {"loss": loss, "preds": preds, "targets": targets}
