from typing import Any, List, Optional

import hydra
import torch
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from ..utils import utils
from .optimizer.scheduler import create_scheduler

log = utils.get_logger(__name__)


class LitClassification(LightningModule):
    """
    LightningModule for classification.
    """

    def __init__(self, cfg: Optional[DictConfig] = None):
        super().__init__()
        self.save_hyperparameters()
        config = cfg
        self.config = config

        # model
        log.info(f"Instantiating module <{config.module._target_}>")
        self.model = hydra.utils.instantiate(
            config.module, num_classes=config.datamodule.num_classes
        )

        # load from checkpoint
        if config.get("load_from_checkpoint"):
            ckpt = torch.load(config.load_from_checkpoint)
            missing_keys, unexpected_keys = self.load_state_dict(ckpt["state_dict"], strict=False)
            log.info(f"[ckpt] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.")
            log.info(f"[ckpt] Load checkpoint from {config.load_from_checkpoint}.")

        # loss function
        log.info(f"Instantiating module <{config.loss._target_}>")
        self.criterion = hydra.utils.instantiate(config.loss)

        # metric
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy()])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # ------------
    # train
    # ------------

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        metric = self.train_metrics(preds, targets)
        self.log("train/loss", loss)
        self.log_dict(metric, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    # ------------
    # validation
    # ------------

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        metric = self.val_metrics(preds, targets)
        self.log("val/loss", loss)
        self.log_dict(metric, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    # ------------
    # test
    # ------------

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        metric = self.val_metrics(preds, targets)
        self.log("test/loss", loss)
        self.log_dict(metric)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    # ------------
    # optim
    # ------------

    def configure_optimizers(self):
        config = self.config

        num_steps_per_epoch = int(
            self.trainer.datamodule.train_len / config.datamodule.effective_batch_size + 0.5
        )
        max_epoch = config.trainer.max_epochs
        max_iterations = max_epoch * num_steps_per_epoch
        sch_times = None

        if config.scheduler.get("warmup"):
            if config.scheduler.policy == "epoch":
                sch_times = max_epoch - config.scheduler.warmup.times
            elif config.scheduler.policy == "iteration":
                if isinstance(config.scheduler.warmup.times, float):
                    sch_times = (
                        max_iterations - config.scheduler.warmup.times * num_steps_per_epoch
                    )
                else:
                    sch_times = max_iterations - config.scheduler.warmup.times
            else:
                raise ValueError(
                    "scheduler_policy should be epoch or iteration,"
                    f"but '{config.scheduler.policy}' given."
                )

        # === Optimizer ===
        log.info(f"Instantiating module <{config.optimizer._target_}>")
        if config.optimizer._target_.split(".")[-1] in ["LARS"]:
            optimizer = hydra.utils.instantiate(config.optimizer, self.model)
        else:
            optimizer = hydra.utils.instantiate(config.optimizer, self.model.parameters())

        # === Scheduler ===
        schedulers = []
        if config.scheduler.get("name"):
            log.info(f"Creating module <{config.scheduler.name}>")
            sch = create_scheduler(optimizer=optimizer, sch_times=sch_times, **config.scheduler)
            schedulers.append(sch)

        return [optimizer], schedulers
