from typing import Any, List, Optional

import einops
import hydra
import torch
import torch.distributed as dist
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


def remove_empty_features(features):
    zero_mask = features.norm(dim=-1) < 1e-12
    features = features[~zero_mask]
    return features


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

        # --- begin:SSL ---
        self._max_superpixel_id = 22000
        self._dim = 256  # hard-coded
        self.ssl = "ssl" in self.config and self.config.ssl.enable
        self.da = self.config.datamodule.src != self.config.datamodule.tgt
        if self.ssl:
            self._init_memqueue()
        # --- end:SSL ---

        # load from checkpoint
        if self.config.load_from_checkpoint:
            self.load_from_checkpoint(checkpoint=self.config.load_from_checkpoint)
            if self.ssl:
                self._update_proto_ready("src")
                self._update_proto_ready("tgt")

    # ------------
    # ssl
    # ------------

    def _create_proto_slot(self, domain):
        for k_, k_rep_ in zip(self.k, self.k_rep):
            self.register_buffer(f"proto_{domain}_{k_}", torch.zeros((k_rep_, k_, self._dim)))

    def _update_proto_ready(self, domain):
        updated = torch.norm(getattr(self, f"proto_{domain}_{self.k[0]}")) > 1e-12
        setattr(self, f"proto_{domain}_ready", updated)

    def _init_memqueue(self):

        # calculate the expanded k_list
        self.k = k = list(self.config.ssl.k)
        self.k_rep = k_rep = list(self.config.ssl.k_rep)
        self.k_list = []
        for k_, k_rep_ in zip(k, k_rep):
            self.k_list += k_rep_ * [k_]

        self.memqueue_tgt = self.memqueue_src = None
        if self.ssl:
            # ssl: memory queue (cpu) on master node
            if self.global_rank == 0:
                self.memqueue_src = MemQueue(name="src")
            # ssl: proto on each node
            self._create_proto_slot("src")
            self.proto_src_ready = False
        if self.ssl and self.da:
            # ssl: memory queue (cpu) on master node
            if self.global_rank == 0:
                self.memqueue_tgt = MemQueue(name="tgt")
            # ssl: proto on each node
            self._create_proto_slot("tgt")
            self.proto_tgt_ready = False

    def _cal_feature_by_superpixel(self, features, labels, superpixels):
        resolution = features.shape[-2:]

        # prepare features
        features = features.detach()
        features = einops.rearrange(features, "b c h w -> (b h w) c")

        # downsample label and superpixel
        labels = downsample_mask(labels, size=resolution)
        superpixels = downsample_mask(superpixels, size=resolution)

        # remove ignore_index
        mask = labels != self.config.datamodule.ignore_index
        labels = labels[mask]
        superpixels = superpixels[mask]
        features = features[mask]

        # mean according to superpixel
        mean_features = torch.zeros((self._max_superpixel_id, self._dim), device=self.device)
        mean_features.scatter_add_(0, superpixels.unsqueeze(1).expand((-1, self._dim)), features)

        # normalize
        mean_features = F.normalize(mean_features, dim=1, p=2, eps=1e-12)

        return mean_features

    def _ssl_loss(self, features, domain):
        if not getattr(self, f"proto_{domain}_ready"):
            return torch.sum(torch.zeros((1)).cuda())
        else:
            return torch.sum(torch.zeros((1)).cuda())

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

        self.log("train/src_cls_loss", src_loss)
        loss = src_loss
        ret = dict(loss=loss)

        # --- begin:SSL ---
        # ssl loss
        if self.ssl:
            src_ssl_loss = self._ssl_loss(src_results["feature"], "src")
            self.log("train/src_ssl_loss", src_ssl_loss)
            loss += src_ssl_loss
        if self.ssl and self.da:
            tgt_results = self.forward(tgt_imgs)

            tgt_ssl_loss = self._ssl_loss(tgt_results["feature"], "tgt")
            self.log("train/tgt_ssl_loss", tgt_ssl_loss)
            loss += tgt_ssl_loss

        # calculate features by superpixel
        if self.ssl:
            src_mean_features = self._cal_feature_by_superpixel(
                src_results["feature"], src_labels, batch_src["superpixel"]
            )
            ret["src_mean_features"] = src_mean_features
        if self.ssl and self.da:
            tgt_mean_features = self._cal_feature_by_superpixel(
                tgt_results["feature"], tgt_labels, batch_tgt["superpixel"]
            )
            ret["tgt_mean_features"] = tgt_mean_features
        # --- end:SSL ---

        # metric
        preds = src_logits.argmax(dim=1)
        preds, labels = strip_ignore_index(preds, src_labels, self.config.datamodule.ignore_index)
        self.train_src_metrics(preds, labels)
        self.log("train/loss", loss)

        return ret

    def training_step_end(self, outputs):

        # --- begin:SSL ---
        def update_mem_and_proto(outputs, domain="src"):
            # (all node) all_gather mean features
            mean_features = outputs.pop(domain + "_mean_features")
            mean_features_list = [
                torch.zeros_like(mean_features) for _ in range(self.trainer.num_gpus)
            ]
            dist.all_gather(mean_features_list, mean_features)
            mean_features_gather = torch.cat(mean_features_list)
            # (master) update memqueue
            if self.global_rank == 0:
                memqueue = getattr(self, "memqueue_" + domain)
                mean_features_to_push = remove_empty_features(mean_features_gather)
                mean_features_to_push = mean_features_to_push.detach().cpu().numpy()
                memqueue.push(mean_features_to_push)
                # (master) compute cluster
                if memqueue.ready:
                    proto_np = memqueue.protos(self.k_list)
                    proto = list_np_to_torch(proto_np)
                    acc = 0
                    for k_, k_rep_ in zip(self.k, self.k_rep):
                        update_value = torch.stack(proto[acc : acc + k_rep_])
                        acc += k_rep_
                        setattr(self, f"proto_{domain}_{k_}", update_value)

            # (all node) broadcast
            for k_ in self.k:
                self_proto = getattr(self, f"proto_{domain}_{k_}")
                dist.broadcast_object_list(self_proto, src=0)
            self._update_proto_ready(domain)

        if self.ssl:
            update_mem_and_proto(outputs, "src")
        if self.ssl and self.da:
            update_mem_and_proto(outputs, "tgt")
        # --- end:SSL ---

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
        imgs, labels, idxs = batch["image"], batch["label"], batch["index"]

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
