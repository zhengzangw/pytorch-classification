import pytest
import torch

from src.modules.deeplab import DeepLab


def test_DeepLab():
    model = DeepLab()
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    assert "feature" in output
    assert "out" in output
