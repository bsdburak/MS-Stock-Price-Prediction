"""Reusable PyTorch training and validation helpers."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

BatchIterable = Iterable[tuple[Tensor, Tensor]]


def _validate_prediction_shape(predictions: Tensor, targets: Tensor) -> None:
    """Prevent silent broadcasting when a model returns the wrong output shape."""
    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction and target shapes must match, "
            f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
        )


def train_one_epoch(
    model: nn.Module,
    data_loader: BatchIterable,
    optimizer: Optimizer,
    loss_function: nn.Module,
    device: torch.device | str = "cpu",
) -> float:
    """Train for one epoch and return sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    sample_count = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        _validate_prediction_shape(predictions, targets)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Cannot train on an empty data loader.")

    return total_loss / sample_count


def evaluate_loss(
    model: nn.Module,
    data_loader: BatchIterable,
    loss_function: nn.Module,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate without gradients and return sample-weighted mean loss."""
    model.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            _validate_prediction_shape(predictions, targets)
            loss = loss_function(predictions, targets)

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    return total_loss / sample_count
