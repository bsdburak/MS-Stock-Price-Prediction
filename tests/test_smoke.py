import torch

from ms_stock_prediction import __version__


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_torch_tensor_creation() -> None:
    tensor = torch.tensor([1.0, 2.0, 3.0])

    assert tensor.shape == (3,)
    assert tensor.sum().item() == 6.0
