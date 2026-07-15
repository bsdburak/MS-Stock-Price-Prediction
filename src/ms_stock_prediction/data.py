"""Download and validate the raw Microsoft stock price dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def default_end_date() -> str:
    """Exclude the current calendar day to avoid a potentially incomplete session."""
    return date.today().isoformat()


@dataclass(frozen=True)
class StockDataConfig:
    """Configuration for the reproducible raw dataset snapshot."""

    ticker: str = "MSFT"
    start: str = "2010-01-01"
    end: str = field(default_factory=default_end_date)
    output_path: Path = Path("data/raw/msft_daily_latest.csv")

    @property
    def metadata_path(self) -> Path:
        """Return the tracked sidecar path for the ignored raw CSV snapshot."""
        return self.output_path.with_suffix(".metadata.json")


def validate_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, ordered frame or raise for an invalid dataset."""
    if data.empty:
        raise ValueError("Downloaded stock data is empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    clean = data.loc[:, REQUIRED_COLUMNS].copy()
    clean.index = pd.DatetimeIndex(pd.to_datetime(clean.index), name="Date")
    if clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)

    if clean.index.has_duplicates:
        raise ValueError("Stock data contains duplicate dates.")

    clean = clean.sort_index()
    clean = clean.apply(pd.to_numeric, errors="raise")

    if clean.isna().any().any():
        raise ValueError("Stock data contains missing values.")
    if (clean.loc[:, PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Stock prices must be positive.")
    if (clean["Volume"] < 0).any():
        raise ValueError("Stock volume cannot be negative.")
    if (clean["High"] < clean["Low"]).any():
        raise ValueError("High price cannot be lower than low price.")

    return clean


def download_stock_data(config: StockDataConfig) -> pd.DataFrame:
    """Download adjusted daily OHLCV data and apply the data contract."""
    data = yf.download(
        tickers=config.ticker,
        start=config.start,
        end=config.end,
        interval="1d",
        auto_adjust=True,
        actions=False,
        prepost=False,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    return validate_stock_data(data)


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_stock_data(data: pd.DataFrame, config: StockDataConfig) -> dict[str, object]:
    """Save the raw CSV and a tracked metadata sidecar describing the snapshot."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.output_path, index=True, date_format="%Y-%m-%d")

    metadata: dict[str, object] = {
        "ticker": config.ticker,
        "source": "Yahoo Finance via yfinance",
        "interval": "1d",
        "auto_adjust": True,
        "start_inclusive": config.start,
        "end_exclusive": config.end,
        "first_observation": data.index.min().date().isoformat(),
        "last_observation": data.index.max().date().isoformat(),
        "rows": len(data),
        "columns": list(data.columns),
        "downloaded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "csv_file": config.output_path.name,
        "csv_bytes": config.output_path.stat().st_size,
        "csv_sha256": file_sha256(config.output_path),
    }
    config.metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the default dataset contract."""
    defaults = StockDataConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=defaults.ticker)
    parser.add_argument("--start", default=defaults.start, help="Inclusive start date.")
    parser.add_argument(
        "--end",
        default=defaults.end,
        help="Exclusive end date; defaults to today to avoid an incomplete session.",
    )
    parser.add_argument("--output", type=Path, default=defaults.output_path)
    return parser.parse_args()


def main() -> None:
    """Download, validate, save, and summarize the configured raw dataset."""
    args = parse_args()
    config = StockDataConfig(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        output_path=args.output,
    )
    data = download_stock_data(config)
    metadata = save_stock_data(data, config)

    print(f"Saved {len(data)} rows to {config.output_path}")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Columns: {', '.join(data.columns)}")
    print(f"Metadata: {config.metadata_path}")
    print(f"SHA-256: {metadata['csv_sha256']}")


if __name__ == "__main__":
    main()
