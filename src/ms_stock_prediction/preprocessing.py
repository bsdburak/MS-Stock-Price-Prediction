"""Reusable preparation pipeline for univariate stock price sequences."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class PreparationConfig:
    """Configuration shared by notebook exploration and model training."""

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    lookback: int = 20
    batch_size: int = 32
    shuffle_seed: int = 42
    feature_range: tuple[float, float] = (-1.0, 1.0)


@dataclass
class PreparedTimeSeries:
    """All leakage-safe artifacts required by recurrent model training."""

    scaler: MinMaxScaler
    train_prices: pd.DataFrame
    validation_prices: pd.DataFrame
    test_prices: pd.DataFrame
    X_train: Tensor
    y_train: Tensor
    X_validation: Tensor
    y_validation: Tensor
    X_test: Tensor
    y_test: Tensor
    train_target_dates: pd.DatetimeIndex
    validation_target_dates: pd.DatetimeIndex
    test_target_dates: pd.DatetimeIndex
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


def load_close_prices(csv_path: Path | str) -> pd.DataFrame:
    """Load and validate a date-indexed Close price frame from the raw CSV."""
    data = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    if "Close" not in data.columns:
        raise ValueError("Raw stock data must contain a Close column.")
    return _validate_close_prices(data[["Close"]])


def _validate_close_prices(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Return a clean chronological Close frame or reject invalid input."""
    if close_prices.empty:
        raise ValueError("Close price data cannot be empty.")
    if list(close_prices.columns) != ["Close"]:
        raise ValueError("Close price data must contain exactly one column named Close.")

    clean = close_prices.copy()
    clean.index = pd.DatetimeIndex(pd.to_datetime(clean.index), name="Date")
    if clean.index.has_duplicates:
        raise ValueError("Close price data contains duplicate dates.")
    clean = clean.sort_index()
    clean["Close"] = pd.to_numeric(clean["Close"], errors="raise")

    if clean["Close"].isna().any():
        raise ValueError("Close price data contains missing values.")
    if (clean["Close"] <= 0).any():
        raise ValueError("Close prices must be positive.")
    return clean


def _validate_config(config: PreparationConfig, observation_count: int) -> None:
    """Reject split and batching settings that cannot produce valid sequences."""
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < config.validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if config.train_ratio + config.validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must leave a non-empty test split.")
    if config.lookback <= 0:
        raise ValueError("lookback must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    train_count = int(observation_count * config.train_ratio)
    validation_count = int(observation_count * config.validation_ratio)
    test_count = observation_count - train_count - validation_count
    if train_count <= config.lookback:
        raise ValueError("Training split must contain more observations than lookback.")
    if validation_count == 0 or test_count == 0:
        raise ValueError("Validation and test splits must be non-empty.")


def _scaled_frame(
    scaler: MinMaxScaler,
    prices: pd.DataFrame,
    *,
    fit: bool,
) -> pd.DataFrame:
    """Scale a frame while preserving its date index."""
    values = scaler.fit_transform(prices) if fit else scaler.transform(prices)
    return pd.DataFrame(values, index=prices.index, columns=["Close_Scaled"])


def _create_sequences(
    scaled_data: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create windows whose split membership is determined by target date."""
    values = scaled_data.to_numpy(dtype=np.float32)
    allowed_targets = set(target_index)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_dates: list[pd.Timestamp] = []

    for target_position in range(lookback, len(scaled_data)):
        target_date = scaled_data.index[target_position]
        if target_date not in allowed_targets:
            continue
        inputs.append(values[target_position - lookback : target_position])
        targets.append(values[target_position])
        target_dates.append(target_date)

    if not inputs:
        raise ValueError("No sequences could be created for the requested target dates.")

    return (
        np.stack(inputs),
        np.stack(targets),
        pd.DatetimeIndex(target_dates, name="Date"),
    )


def prepare_time_series(
    close_prices: pd.DataFrame,
    config: PreparationConfig | None = None,
) -> PreparedTimeSeries:
    """Create chronological splits, scaled windows, tensors, and data loaders."""
    config = config or PreparationConfig()
    close_prices = _validate_close_prices(close_prices)
    _validate_config(config, len(close_prices))

    train_end = int(len(close_prices) * config.train_ratio)
    validation_end = train_end + int(len(close_prices) * config.validation_ratio)
    train_prices = close_prices.iloc[:train_end].copy()
    validation_prices = close_prices.iloc[train_end:validation_end].copy()
    test_prices = close_prices.iloc[validation_end:].copy()

    scaler = MinMaxScaler(feature_range=config.feature_range)
    train_scaled = _scaled_frame(scaler, train_prices, fit=True)
    validation_scaled = _scaled_frame(scaler, validation_prices, fit=False)
    test_scaled = _scaled_frame(scaler, test_prices, fit=False)
    full_scaled = pd.concat([train_scaled, validation_scaled, test_scaled])

    X_train_array, y_train_array, train_target_dates = _create_sequences(
        full_scaled, train_prices.index, config.lookback
    )
    X_validation_array, y_validation_array, validation_target_dates = _create_sequences(
        full_scaled, validation_prices.index, config.lookback
    )
    X_test_array, y_test_array, test_target_dates = _create_sequences(
        full_scaled, test_prices.index, config.lookback
    )

    X_train = torch.from_numpy(X_train_array)
    y_train = torch.from_numpy(y_train_array)
    X_validation = torch.from_numpy(X_validation_array)
    y_validation = torch.from_numpy(y_validation_array)
    X_test = torch.from_numpy(X_test_array)
    y_test = torch.from_numpy(y_test_array)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.shuffle_seed),
    )
    validation_loader = DataLoader(
        TensorDataset(X_validation, y_validation),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return PreparedTimeSeries(
        scaler=scaler,
        train_prices=train_prices,
        validation_prices=validation_prices,
        test_prices=test_prices,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        train_target_dates=train_target_dates,
        validation_target_dates=validation_target_dates,
        test_target_dates=test_target_dates,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )
