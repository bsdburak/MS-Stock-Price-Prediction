import json
from pathlib import Path

import pandas as pd
import pytest

from ms_stock_prediction import data as data_module
from ms_stock_prediction.data import (
    REQUIRED_COLUMNS,
    StockDataConfig,
    download_stock_data,
    save_stock_data,
    validate_stock_data,
)


def sample_stock_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [102.0, 100.0],
            "High": [104.0, 103.0],
            "Low": [101.0, 99.0],
            "Close": [103.0, 102.0],
            "Volume": [1_200_000, 1_000_000],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-02"]),
    )


def test_validate_stock_data_orders_dates_and_columns() -> None:
    clean = validate_stock_data(sample_stock_data())

    assert clean.index.is_monotonic_increasing
    assert clean.index.name == "Date"
    assert tuple(clean.columns) == REQUIRED_COLUMNS


def test_validate_stock_data_rejects_missing_columns() -> None:
    invalid = sample_stock_data().drop(columns="Volume")

    with pytest.raises(ValueError, match="Missing required columns: Volume"):
        validate_stock_data(invalid)


def test_download_stock_data_uses_reproducible_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    def fake_download(**kwargs: object) -> pd.DataFrame:
        received.update(kwargs)
        return sample_stock_data()

    monkeypatch.setattr(data_module.yf, "download", fake_download)
    config = StockDataConfig()

    result = download_stock_data(config)

    assert not result.empty
    assert received["tickers"] == "MSFT"
    assert received["start"] == "2010-01-01"
    assert received["end"] == config.end
    assert received["auto_adjust"] is True
    assert received["multi_level_index"] is False


def test_save_stock_data_writes_date_index(tmp_path: Path) -> None:
    output_path = tmp_path / "stock.csv"
    config = StockDataConfig(end="2024-01-04", output_path=output_path)
    clean = validate_stock_data(sample_stock_data())

    metadata = save_stock_data(clean, config)
    loaded = pd.read_csv(output_path)
    saved_metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert config.metadata_path.exists()
    assert loaded.columns[0] == "Date"
    assert len(loaded) == len(clean)
    assert metadata["rows"] == len(clean)
    assert saved_metadata["csv_sha256"] == metadata["csv_sha256"]
