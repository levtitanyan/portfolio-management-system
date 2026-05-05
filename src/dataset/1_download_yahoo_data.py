"""
Download daily historical market data from Yahoo Finance with yfinance.

Downloads 30 large U.S. stocks across sectors, plus market reference ETFs
and VIX (volatility index). Saves one CSV per ticker in data/raw/.

Run from the project root:
    python src/dataset/1_download_yahoo_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from universes import get_all_download_tickers, get_data_dir

START_DATE = "2015-01-01"
END_DATE   = "2026-04-01"
OUTPUT_DIR = get_data_dir() / "raw"
TICKERS    = get_all_download_tickers()

CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def sanitize_filename(ticker: str) -> str:
    """
    Remove characters that are unsafe in filenames across operating systems.
    Example: ^VIX becomes VIX.
    """
    return ticker.replace("^", "")


def normalize_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flatten yfinance MultiIndex output into a single flat column set.
    yfinance returns either (field, ticker) or (ticker, field) depending
    on the version — this handles both cases.
    """
    if isinstance(data.columns, pd.MultiIndex):
        first_level = set(data.columns.get_level_values(0))

        if ticker in first_level and len(first_level) == 1:
            data = data[ticker].copy()
        else:
            data.columns = data.columns.get_level_values(0)

    return data


def prepare_dataframe(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize raw yfinance output into a clean, consistent DataFrame.

    - Handles missing Adj Close by falling back to Close.
    - Ensures Date column is named correctly regardless of yfinance version.
    - Returns only the expected CSV columns in the correct order.
    """
    data = normalize_columns(data, ticker)

    if "Adj Close" not in data.columns:
        if "Close" not in data.columns:
            raise ValueError(
                f"[{ticker}] Downloaded data is missing both 'Close' and 'Adj Close'."
            )
        data["Adj Close"] = data["Close"]

    data = data.reset_index()

    if "Date" not in data.columns:
        first_column = data.columns[0]
        data = data.rename(columns={first_column: "Date"})

    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

    missing_columns = [c for c in CSV_COLUMNS if c not in data.columns]
    if missing_columns:
        raise ValueError(
            f"[{ticker}] Missing expected columns after preparation: {missing_columns}"
        )

    return data[CSV_COLUMNS]


def download_ticker(ticker: str, output_dir: Path) -> tuple[str, int]:
    """
    Download one ticker, save it as a CSV, and return (filename, row_count).
    The filename is sanitized so that tickers like ^VIX are saved as VIX.csv.
    """
    data = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )

    if data.empty:
        raise ValueError(f"[{ticker}] No data returned from Yahoo Finance.")

    cleaned_data = prepare_dataframe(data, ticker)

    filename = sanitize_filename(ticker) + ".csv"
    output_path = output_dir / filename
    cleaned_data.to_csv(output_path, index=False)

    return filename, len(cleaned_data)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful: list[tuple[str, str, int]] = []
    failed: list[tuple[str, str]] = []

    for ticker in TICKERS:
        try:
            filename, row_count = download_ticker(ticker, OUTPUT_DIR)
            successful.append((ticker, filename, row_count))
            print(f"  OK  {ticker:<6} -> {filename}  ({row_count} rows)")
        except Exception as error:
            failed.append((ticker, str(error)))
            print(f"  FAIL {ticker:<6} -> {error}")

    print("\n── Download summary ──────────────────────────────────")
    print(f"Successful : {len(successful)} / {len(TICKERS)}")
    for ticker, filename, rows in successful:
        print(f"  {ticker:<6} {filename:<12} {rows} rows")

    if failed:
        print(f"\nFailed : {len(failed)}")
        for ticker, error_message in failed:
            print(f"  {ticker:<6} {error_message}")

    print(f"\nFiles saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
