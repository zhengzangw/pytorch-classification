from typing import Any, List, Optional

import einops
import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig
from torch_scatter import scatter_max

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


def list_np_to_torch(np_list, device=None):
    ret = []
    for np_array in np_list:
        ret.append(torch.from_numpy(np_array).to(device))
    return ret


def remove_empty_features(features, labels=None):
    zero_mask = features.norm(dim=-1) < 1e-12
    features = features[~zero_mask]
    ret = dict(features=features)
    if labels is not None:
        labels = labels[~zero_mask]
        ret["labels"] = labels
    return ret


def rev_domain(domain):
    return dict(src="tgt", tgt="src")[domain]


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
        if "load_from_checkpoint" in self.config and self.config.load_from_checkpoint:
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
                self.memqueue_src = MemQueue(
                    name="src",
                    size=self.config.ssl.queue_size,
                    num_classes=self.config.datamodule.num_classes,
                )
            # ssl: proto on each node
            self._create_proto_slot("src")
            self.proto_src_ready = False
        if self.ssl and self.da:
            # ssl: memory queue (cpu) on master node
            if self.global_rank == 0:
                self.memqueue_tgt = MemQueue(
                    name="tgt",
                    size=self.config.ssl.queue_size,
                    num_classes=self.config.datamodule.num_classes,
                )
            # ssl: proto on each node
            self._create_proto_slot("tgt")
            self.proto_tgt_ready = False

    def _cal_feature_by_superpixel(
        self, features, superpixels, labels=None, remove_ignore_index=True
    ):
        resolution = features.shape[-2:]

        # prepare features
        features = features.detach()
        features = einops.rearrange(features, "b c h w -> (b h w) c")

        # downsample label and superpixel
        superpixels = downsample_mask(superpixels, size=resolution)
        if labels is not None:
            labels = downsample_mask(labels, size=resolution)

        # remove ignore_index
        if remove_ignore_index:
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

    def _batch_contrast_loss(self, batch_features, batch_anchors, phi=0.05, eps=1e-12):
        # batch_features: [B x dim]
        # anchors: [n_group x k x dim]
        group_logits_T = batch_anchors @ batch_features.T
        # [n_group x k x B]
        group_logits_scale_T = group_logits_T / phi
        group_prob_T = F.softmax(group_logits_scale_T, dim=1)
        entropy = -torch.sum(group_prob_T * torch.log(group_prob_T + eps), dim=1)
        loss = torch.mean(entropy)
        return loss

    def _ssl_loss(self, features, domain):
        loss = torch.zeros(()).to(self.device)
        if not self.proto_src_ready or not self.proto_tgt_ready:
            return loss

        # clean features
        features = F.normalize(features, dim=1)
        features = remove_empty_features(features)["features"]

        # centroids [k_rep x k x dim]
        for k_, k_rep_ in zip(self.k, self.k_rep):
            # in-domain
            centroids = getattr(self, f"proto_{domain}_{k_}")
            loss += self._batch_contrast_loss(features, centroids, phi=self.config.ssl.phi)
            # cross-domain
            centroids = getattr(self, f"proto_{rev_domain(domain)}_{k_}")
            loss += self._batch_contrast_loss(features, centroids)
        return loss

    def _classify(self, features):
        weight = self.model.decoder.classifier.weight.data.detach().flatten(start_dim=1).T
        return (features @ weight).argmax(dim=-1)

    # ------------
    # train
    # ------------

    def training_step(self, batch: Any, batch_idx: int):
        loss = torch.zeros(()).to(self.device)
        ret = dict(loss=loss)

        # Get src info
        batch_src = batch["src"]
        src_imgs, src_labels = (
            batch_src["image"],
            batch_src["label"],
        )

        # Get tgt info
        if "tgt" in batch:
            batch_tgt = batch["tgt"]
            tgt_imgs = batch_tgt["image"]

        # loss
        src_results = self.forward(src_imgs)
        src_logits = src_results["out"]
        src_loss = self.criterion(src_logits, src_labels)

        self.log("train/src_cls", src_loss, prog_bar=True)
        loss += src_loss

        # --- begin:SSL ---
        # calculate tgt features
        if self.ssl and self.da:
            tgt_results = self.forward(tgt_imgs)

        # calculate features by superpixel
        if self.ssl:
            src_mean_features = self._cal_feature_by_superpixel(
                src_results["feature"], batch_src["superpixel"], labels=src_labels
            )
            ret["src_mean_features"] = src_mean_features
            if self.config.ssl.classwise_sample:
                ret["src_mean_labels"] = self._classify(src_mean_features)
        if self.ssl and self.da:
            tgt_mean_features = self._cal_feature_by_superpixel(
                tgt_results["feature"],
                batch_tgt["superpixel"],
                remove_ignore_index=False,
            )
            ret["tgt_mean_features"] = tgt_mean_features
            if self.config.ssl.classwise_sample:
                ret["tgt_mean_labels"] = self._classify(tgt_mean_features)

        # calculate loss
        if self.ssl:
            src_ssl_loss = self._ssl_loss(src_mean_features, "src")
            self.log("train/src_ssl", src_ssl_loss, prog_bar=True)
            loss += src_ssl_loss
        if self.ssl and self.da:
            tgt_ssl_loss = self._ssl_loss(tgt_mean_features, "tgt")
            self.log("train/tgt_ssl", tgt_ssl_loss, prog_bar=True)
            loss += tgt_ssl_loss
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

            # (all node) all_gather mean labels
            if self.config.ssl.classwise_sample:
                mean_labels = outputs.pop(domain + "_mean_labels")
                mean_labels_list = [
                    torch.zeros_like(mean_labels) for _ in range(self.trainer.num_gpus)
                ]
                dist.all_gather(mean_labels_list, mean_labels)
                mean_labels_gather = torch.cat(mean_labels_list)
            else:
                mean_labels_gather = None

            # (master) update memqueue
            if self.global_rank == 0:
                memqueue = getattr(self, "memqueue_" + domain)
                ret = remove_empty_features(mean_features_gather, labels=mean_labels_gather)
                mean_features_to_push = ret["features"].detach().cpu().numpy()
                if self.config.ssl.classwise_sample:
                    mean_labels_to_push = ret["labels"].detach().cpu().numpy()
                else:
                    mean_labels_to_push = None
                memqueue.push(
                    mean_features_to_push,
                    sample_ratio=self.config.ssl.sample_ratio,
                    labels=mean_labels_to_push,
                    classwise_sample=self.config.ssl.classwise_sample,
                )

                # (master) compute cluster
                if memqueue.ready:
                    proto_np = memqueue.protos(self.k_list)
                    proto = list_np_to_torch(proto_np, device=self.device)
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
