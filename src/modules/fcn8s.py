from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torchvision import models


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):
    def __init__(self, num_classes=19, pretrained_backbone=True, pretrained=False, **kwargs):
        super(FCN8s, self).__init__()
        self.pretrained = pretrained_backbone

        # vgg16
        vgg = models.vgg16(pretrained=pretrained)
        features, classifier = (
            list(vgg.features.children()),
            list(vgg.classifier.children()),
        )
        for f in features:
            if "MaxPool" in f.__class__.__name__:
                f.ceil_mode = True
            elif "ReLU" in f.__class__.__name__:
                f.inplace = True

        # conv1
        features[0].padding = (100, 100)
        self.features1 = nn.Sequential(*features[:5])
        # conv2, conv3
        self.features3 = nn.Sequential(*features[5:17])
        # conv4
        self.features4 = nn.Sequential(*features[17:24])
        # conv5
        self.features5 = nn.Sequential(*features[24:])

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # score_fr
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, bias=False
        )

        self._initialize_weights(classifier)

    def _initialize_weights(self, classifier):
        if self.pretrained:
            # fc6
            self.fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
            self.fc6.bias.data.copy_(classifier[0].bias.data)
            # fc7
            self.fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
            self.fc7.bias.data.copy_(classifier[3].bias.data)
            # conv
            for m_name in ["score_fr", "score_pool3", "score_pool4"]:
                m = getattr(self, m_name)
                m.weight.data.zero_()
                m.bias.data.zero_()
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.zero_()
                    if m.bias is not None:
                        m.bias.data.zero_()
        # convtranspose
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0]
                )
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        x_size = x.size()

        h = x
        h = self.features1(h)
        h = self.features3(h)
        pool3 = h  # 1/8
        h = self.features4(h)
        pool4 = h  # 1/16
        h = self.features5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5 : (5 + upscore2.size()[2]), 5 : (5 + upscore2.size()[3])]
        h = self.upscore_pool4(h + upscore2)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[
            :,
            :,
            9 : (9 + upscore_pool4.size()[2]),
            9 : (9 + upscore_pool4.size()[3]),
        ]

        h = self.upscore8(h + upscore_pool4)
        h = h[:, :, 31 : (31 + x.size()[2]), 31 : (31 + x.size()[3])].contiguous()

        result = OrderedDict()
        result["out"] = h

        return result


if __name__ == "__main__":
    fcn = FCN8s(19)
