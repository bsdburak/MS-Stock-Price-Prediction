import math

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from ms_stock_prediction.training import evaluate_loss, train_one_epoch


def make_linear_loader(batch_size: int = 4) -> DataLoader:
    inputs = torch.linspace(-1, 1, steps=20).reshape(-1, 1)
    targets = 2 * inputs + 1
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)


def test_train_one_epoch_updates_parameters_and_reduces_loss() -> None:
    model = nn.Linear(1, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    loader = make_linear_loader()
    loss_function = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    initial_loss = evaluate_loss(model, loader, loss_function)
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    train_loss = train_one_epoch(model, loader, optimizer, loss_function)
    final_loss = evaluate_loss(model, loader, loss_function)

    assert math.isfinite(train_loss)
    assert final_loss < initial_loss
    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_parameters, model.parameters(), strict=True)
    )


def test_evaluate_loss_does_not_update_parameters() -> None:
    model = nn.Linear(1, 1)
    loader = make_linear_loader()
    loss_function = nn.MSELoss()
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    loss = evaluate_loss(model, loader, loss_function)

    assert math.isfinite(loss)
    assert not model.training
    assert all(
        torch.equal(before, after)
        for before, after in zip(original_parameters, model.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluate_loss_uses_sample_weighted_mean() -> None:
    inputs = torch.tensor([[1.0], [1.0], [3.0]])
    targets = torch.zeros_like(inputs)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)

    loss = evaluate_loss(nn.Identity(), loader, nn.MSELoss())

    assert loss == pytest.approx(11 / 3)


def test_empty_loader_is_rejected() -> None:
    inputs = torch.empty((0, 1))
    targets = torch.empty((0, 1))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = nn.Linear(1, 1)
    loss_function = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="Cannot train on an empty data loader"):
        train_one_epoch(model, loader, optimizer, loss_function)
    with pytest.raises(ValueError, match="Cannot evaluate an empty data loader"):
        evaluate_loss(model, loader, loss_function)
