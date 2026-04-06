"""
Download daily historical market data from Yahoo Finance with yfinance.

"""

from pathlib import Path

import pandas as pd
import yfinance as yf


START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
OUTPUT_DIR = Path("data/raw")

TICKERS = [
    "AAPL", # Apple
    "MSFT", # Microsoft
    "JPM",  # JPMorgan Chase
    "GS",   # Goldman Sachs
    "V",    # Visa
    "WMT",  # Walmart
    "HD",   # Home Depot
    "MCD",  # McDonald's
    "JNJ",  # Johnson & Johnson
    "PG",   # Procter & Gamble
    "KO",   # Coca-Cola
    "CVX",  # Chevron
    "CAT",  # Caterpillar
    "IBM",  # IBM
    "DIS",  # Disney
    "SPY",  # SPDR S&P 500 ETF
    "^VIX", # CBOE Volatility Index
]

CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def normalize_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flatten yfinance output so we can work with one clean set of columns.
    """
    if isinstance(data.columns, pd.MultiIndex):
        first_level = set(data.columns.get_level_values(0))

        # Some yfinance versions return (ticker, field).
        if ticker in first_level and len(first_level) == 1:
            data = data[ticker].copy()
        else:
            # Other versions return (field, ticker).
            data.columns = data.columns.get_level_values(0)

    return data


def prepare_dataframe(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Keep the CSV beginner-friendly and stable across yfinance versions.
    """
    data = normalize_columns(data, ticker)

    # Newer yfinance behavior can omit "Adj Close" in some cases.
    # If that happens, keep the output schema consistent by falling back to Close.
    if "Adj Close" not in data.columns:
        if "Close" not in data.columns:
            raise ValueError("Downloaded data is missing both 'Close' and 'Adj Close'.")
        data["Adj Close"] = data["Close"]

    data = data.reset_index()

    # After reset_index(), the date column is usually named "Date".
    # This fallback keeps the script resilient if the label changes.
    if "Date" not in data.columns:
        first_column = data.columns[0]
        data = data.rename(columns={first_column: "Date"})

    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

    missing_columns = [column for column in CSV_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    return data[CSV_COLUMNS]


def download_ticker(ticker: str, output_dir: Path) -> int:
    """
    Download one ticker and save it as a CSV file.

    Returns the number of rows written.
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
        raise ValueError("No data returned.")

    cleaned_data = prepare_dataframe(data, ticker)
    output_file = output_dir / f"{ticker}.csv"
    cleaned_data.to_csv(output_file, index=False)

    return len(cleaned_data)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful_downloads = []
    failed_downloads = []

    for ticker in TICKERS:
        try:
            row_count = download_ticker(ticker, OUTPUT_DIR)
            successful_downloads.append((ticker, row_count))
            print(f"Downloaded {ticker} successfully: {row_count} rows")
        except Exception as error:
            failed_downloads.append((ticker, str(error)))
            print(f"Failed to download {ticker}: {error}")

    print("\nDownload summary")
    print(f"Successful downloads: {len(successful_downloads)}")
    for ticker, row_count in successful_downloads:
        print(f"  - {ticker}: {row_count} rows")

    print(f"Failed downloads: {len(failed_downloads)}")
    for ticker, error_message in failed_downloads:
        print(f"  - {ticker}: {error_message}")


if __name__ == "__main__":
    main()
