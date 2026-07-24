import pytest
import torch
from torch import nn

from ms_stock_prediction.models import GRUModel, LSTMModel, count_trainable_parameters


@pytest.mark.parametrize("model_class", [LSTMModel, GRUModel])
def test_recurrent_model_forward_shape(model_class: type[nn.Module]) -> None:
    torch.manual_seed(42)
    model = model_class(input_size=1, hidden_size=32, num_layers=2, output_size=1)
    inputs = torch.randn(8, 20, 1)

    predictions = model(inputs)

    assert predictions.shape == (8, 1)
    assert predictions.dtype == torch.float32
    assert torch.isfinite(predictions).all()


@pytest.mark.parametrize("model_class", [LSTMModel, GRUModel])
def test_recurrent_model_supports_backpropagation(model_class: type[nn.Module]) -> None:
    torch.manual_seed(42)
    model = model_class()
    inputs = torch.randn(4, 20, 1)
    targets = torch.randn(4, 1)

    loss = nn.MSELoss()(model(inputs), targets)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_default_parameter_counts() -> None:
    lstm = LSTMModel()
    gru = GRUModel()

    assert count_trainable_parameters(lstm) == 12_961
    assert count_trainable_parameters(gru) == 9_729
    assert count_trainable_parameters(gru) < count_trainable_parameters(lstm)


@pytest.mark.parametrize("model_class", [LSTMModel, GRUModel])
def test_invalid_hidden_size_is_rejected(model_class: type[nn.Module]) -> None:
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        model_class(hidden_size=0)
