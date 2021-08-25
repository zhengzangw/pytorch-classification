import torch


def strip_ignore_index(preds, label, ignore_index):
    assert label.dtype == torch.long
    mask = label != ignore_index
    label = label[mask]
    preds = preds[mask]
    return preds, label


def set_bn_momentum(model, momentum=0.1):
    # https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/0c67dce524b2eb94dc3587ff2832e28f11440cae/utils/utils.py#L26
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = momentum
