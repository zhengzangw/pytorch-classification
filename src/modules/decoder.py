import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import utils
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

log = utils.get_logger(__name__)


class CosineConv2d(nn.Conv2d):
    # TODO: wrong normalize
    # TODO: normalized class average representation

    def __init__(self, *args, **kwargs):
        log.info("[Decoder] Use cosine classifier.")
        super().__init__(*args, **kwargs, bias=False)

    def forward(self, x):
        # Normalize weight
        weight = F.normalize(self.weight, p=2, eps=1e-12, dim=1)
        # weight = self.weight
        # norm = torch.norm(weight, dim=1)

        # Normalize features
        # x [N, C, H, W]
        x = F.normalize(x, dim=1, p=2, eps=1e-12)

        # Convolution
        # x = super().forward(x)
        x = F.conv2d(x, weight, bias=None, stride=self.stride)

        # x = x / norm

        return x


class ConsineLinear(nn.Module):
    def __init__(self, in_channel, out_channel, *args, **kwargs):
        log.info("[Decoder] Use linear cosine classifier.")
        super().__init__()

        self.fc = nn.Linear(in_channel, out_channel, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.transpose(0, 1)
        x = x.flatten(1)
        x = x.transpose(0, 1)  # [N*H*W, C]

        # breakpoint()
        # self.normalize()
        weight = F.normalize(self.fc.weight, dim=1, p=2, eps=1e-12)
        x = F.normalize(x, dim=1, p=2, eps=1e-12)
        # x = self.fc(x)  # N*H*W, C
        x = F.linear(x, weight, bias=None)

        x = x.reshape(n, h, w, -1)
        x = x.transpose(-1, -2)
        x = x.transpose(-2, -3)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone,
        BatchNorm,
        cosine=False,
        finetune_classifier=False,
        aux=True,
        temparature=1.0,
    ):
        super(Decoder, self).__init__()
        if backbone == "resnet" or backbone == "drn":
            low_level_inplanes = 256
        elif backbone == "xception":
            low_level_inplanes = 128
        elif backbone == "mobilenet":
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.temparature = temparature
        self.cosine = cosine
        self.finetune_classifier = finetune_classifier
        self.aux = aux
        if finetune_classifier:
            log.info("Detach Backbone.")
        if cosine:
            LastLayerConv2d = CosineConv2d
        else:
            LastLayerConv2d = nn.Conv2d
        self.classifier = LastLayerConv2d(256, num_classes, kernel_size=1, stride=1)

        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        if self.finetune_classifier:
            x = x.detach()
        features = x

        x = self.classifier(x)
        x = x / self.temparature
        if self.aux:
            return x, features
        else:
            return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.cosine:
            torch.nn.init.orthogonal_(self.classifier.weight)


def build_decoder(num_classes, backbone, BatchNorm, **kwargs):
    return Decoder(num_classes, backbone, BatchNorm, **kwargs)
