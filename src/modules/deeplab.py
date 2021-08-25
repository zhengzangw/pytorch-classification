# https://github.com/jfzhang95/pytorch-deeplab-xception

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspp import build_aspp
from .backbone import build_backbone
from .decoder import build_decoder
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepLab(nn.Module):
    def __init__(
        self,
        backbone="resnet",
        output_stride=16,
        num_classes=21,
        sync_bn=False,
        is_freeze_bn=False,
        cosine=False,
        finetune_classifier=False,
        aux=True,
        temparature=1.0,
        **kwargs
    ):
        super(DeepLab, self).__init__()
        if backbone == "drn":
            output_stride = 8

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(
            num_classes,
            backbone,
            BatchNorm,
            cosine=cosine,
            finetune_classifier=finetune_classifier,
            aux=aux,
            temparature=temparature,
        )

        self.is_freeze_bn = is_freeze_bn
        self.aux = aux

    def forward(self, input):
        result = OrderedDict()

        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        if self.aux:
            x, features = self.decoder(x, low_level_feat)
            result["feature"] = features
        else:
            x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        result["out"] = x
        return result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if (
                        isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)
                    ):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if (
                        isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)
                    ):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
