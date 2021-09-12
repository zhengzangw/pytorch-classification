from typing import Any, List, Optional

import hydra
import torch
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from ..optimizer.scheduler import create_scheduler
from ..utils import utils
from ..utils.misc import mixup_data

log = utils.get_logger(__name__)


class LitBase(LightningModule):
    def __init__(self, cfg: Optional[DictConfig] = None, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        config = cfg
        self.config = config

        # model
        log.info(f"Instantiating module <{config.module._target_}>")
        self.model = hydra.utils.instantiate(
            config.module, num_classes=config.datamodule.num_classes
        )

        # loss function
        log.info(f"Instantiating module <{config.loss._target_}>")
        self.criterion = hydra.utils.instantiate(config.loss)

    def load_from_checkpoint(self, checkpoint):
        log.info(f"[ckpt] Load checkpoint from {checkpoint}.")
        ckpt = torch.load(checkpoint)
        missing_keys, unexpected_keys = self.load_state_dict(ckpt["state_dict"], strict=False)
        log.info(f"[ckpt] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    # ------------
    # train
    # ------------

    def training_epoch_end(self, outputs: List[Any]):
        pass

    # ------------
    # validation
    # ------------

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    # ------------
    # test
    # ------------

    def test_epoch_end(self, outputs: List[Any]):
        pass

    # ------------
    # optim
    # ------------

    def configure_scheduler(self, optimizer):
        config = self.config
        num_steps_per_epoch = int(
            self.trainer.datamodule.train_len / config.datamodule.effective_batch_size + 0.5
        )
        max_epoch = config.trainer.max_epochs
        max_iterations = max_epoch * num_steps_per_epoch

        if config.scheduler.policy == "epoch":
            sch_times = max_epoch
        else:
            sch_times = max_iterations

        if config.scheduler.get("warmup"):
            if config.scheduler.policy == "epoch":
                sch_times -= config.scheduler.warmup.times
            elif config.scheduler.policy == "iteration":
                if isinstance(config.scheduler.warmup.times, float):
                    sch_times -= config.scheduler.warmup.times * num_steps_per_epoch
                else:
                    sch_times -= config.scheduler.warmup.times
            else:
                raise ValueError(
                    "scheduler_policy should be epoch or iteration,"
                    f"but '{config.scheduler.policy}' given."
                )

        schedulers = []
        if config.scheduler.get("name"):
            log.info(f"Creating module <{config.scheduler.name}>")
            sch = create_scheduler(optimizer=optimizer, sch_times=sch_times, **config.scheduler)
            schedulers.append(sch)

        return schedulers

    def configure_optimizers(self):
        config = self.config

        # === Optimizer ===
        log.info(f"Instantiating module <{config.optimizer._target_}>")
        if config.optimizer._target_.split(".")[-1] in ["LARS"]:
            optimizer = hydra.utils.instantiate(config.optimizer, self.model)
        else:
            optimizer = hydra.utils.instantiate(config.optimizer, self.model.parameters())

        # === Scheduler ===
        schedulers = self.configure_scheduler(optimizer)

        return [optimizer], schedulers
