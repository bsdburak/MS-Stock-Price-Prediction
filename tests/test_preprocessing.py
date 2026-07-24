import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from ms_stock_prediction.preprocessing import PreparationConfig, prepare_time_series


def make_close_prices(observation_count: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": np.linspace(10.0, 200.0, observation_count)},
        index=pd.bdate_range("2020-01-01", periods=observation_count, name="Date"),
    )


def test_prepare_time_series_shapes_and_boundaries() -> None:
    prices = make_close_prices()

    prepared = prepare_time_series(prices)

    assert prepared.X_train.shape == (50, 20, 1)
    assert prepared.y_train.shape == (50, 1)
    assert prepared.X_validation.shape == (15, 20, 1)
    assert prepared.y_validation.shape == (15, 1)
    assert prepared.X_test.shape == (15, 20, 1)
    assert prepared.y_test.shape == (15, 1)
    assert prepared.train_target_dates[0] == prices.index[20]
    assert prepared.validation_target_dates[0] == prices.index[70]
    assert prepared.test_target_dates[0] == prices.index[85]


def test_scaler_fits_only_training_prices() -> None:
    prices = make_close_prices()

    prepared = prepare_time_series(prices)

    assert prepared.scaler.n_samples_seen_ == 70
    np.testing.assert_allclose(prepared.scaler.data_min_, prepared.train_prices.min())
    np.testing.assert_allclose(prepared.scaler.data_max_, prepared.train_prices.max())
    assert prepared.X_validation[0, -1, 0] == pytest.approx(1.0)
    assert prepared.y_validation[0, 0] > 1
    assert prepared.y_test[0, 0] > 1


def test_data_loaders_use_expected_samplers_and_batches() -> None:
    config = PreparationConfig(batch_size=8, shuffle_seed=42)

    prepared = prepare_time_series(make_close_prices(), config)

    assert isinstance(prepared.train_loader.sampler, RandomSampler)
    assert isinstance(prepared.validation_loader.sampler, SequentialSampler)
    assert isinstance(prepared.test_loader.sampler, SequentialSampler)
    validation_X, validation_y = next(iter(prepared.validation_loader))
    torch.testing.assert_close(validation_X, prepared.X_validation[:8])
    torch.testing.assert_close(validation_y, prepared.y_validation[:8])


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (PreparationConfig(train_ratio=0), "train_ratio must be between"),
        (PreparationConfig(validation_ratio=0), "validation_ratio must be between"),
        (
            PreparationConfig(train_ratio=0.8, validation_ratio=0.2),
            "must leave a non-empty test split",
        ),
        (PreparationConfig(lookback=0), "lookback must be positive"),
        (PreparationConfig(batch_size=0), "batch_size must be positive"),
    ],
)
def test_invalid_preparation_config_is_rejected(config: PreparationConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_time_series(make_close_prices(), config)


def test_duplicate_dates_are_rejected() -> None:
    prices = make_close_prices()
    prices.index = pd.DatetimeIndex([prices.index[0], *prices.index[:-1]], name="Date")

    with pytest.raises(ValueError, match="duplicate dates"):
        prepare_time_series(prices)
