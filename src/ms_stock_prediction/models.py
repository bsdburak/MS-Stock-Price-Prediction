"""PyTorch recurrent models for next-day stock price prediction."""

from torch import Tensor, nn


def _validate_model_dimensions(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    output_size: int,
    dropout: float,
) -> None:
    """Reject invalid architecture settings before PyTorch constructs the model."""
    dimensions = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "output_size": output_size,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if not 0 <= dropout < 1:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")


class LSTMModel(nn.Module):
    """Many-to-one LSTM regressor using the final sequence output."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_model_dimensions(input_size, hidden_size, num_layers, output_size, dropout)

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """Map ``[batch, time, feature]`` inputs to ``[batch, output]`` predictions."""
        sequence_outputs, _ = self.recurrent(inputs)
        return self.output(sequence_outputs[:, -1, :])


class GRUModel(nn.Module):
    """Many-to-one GRU regressor using the final sequence output."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_model_dimensions(input_size, hidden_size, num_layers, output_size, dropout)

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """Map ``[batch, time, feature]`` inputs to ``[batch, output]`` predictions."""
        sequence_outputs, _ = self.recurrent(inputs)
        return self.output(sequence_outputs[:, -1, :])


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of model parameters updated during training."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
