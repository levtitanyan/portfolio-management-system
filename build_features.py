"""
Build a clean feature engineering dataset for daily stock return prediction.

This script reads raw stock CSV files from data/raw/, creates technical and
market features, builds a next-day return target, and saves both per-stock
processed files and one combined modeling dataset.

Install dependencies with:
    pip install pandas numpy ta
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
except ImportError as error:
    raise SystemExit(
        "Missing dependency 'ta'. Install dependencies with: pip install pandas numpy ta"
    ) from error


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
FINAL_DATASET_PATH = Path("data/final_model_dataset.csv")
SPY_FILE = "SPY.csv"
VIX_FILE = "^VIX.csv"

OUTPUT_COLUMNS = [
    "Date",
    "ticker",
    "adj_close",
    "volume",
    "log_return",
    "volume_change",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "volatility_10",
    "spy_log_return",
    "vix_close",
    "vix_log_return",
    "target_next_day_return",
]


def load_price_data(csv_path: Path) -> pd.DataFrame:
    """
    Read one CSV file, parse Date as datetime, and sort by Date.
    """
    dataframe = pd.read_csv(csv_path)

    required_columns = {"Date", "Adj Close", "Volume"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    if dataframe["Date"].isna().any():
        raise ValueError("Date column contains invalid values.")

    dataframe = dataframe.sort_values("Date").reset_index(drop=True)
    return dataframe


def build_spy_features(spy_path: Path) -> pd.DataFrame:
    """
    Create market return features from SPY.
    """
    spy_data = load_price_data(spy_path).copy()

    # We use Adj Close because it adjusts for splits and dividends, which makes
    # return calculations more consistent over time than raw Close prices.
    spy_price = spy_data["Adj Close"].astype(float)

    # This is the market return proxy for the overall US market.
    # It helps the model see whether the whole market moved up or down that day.
    spy_data["spy_log_return"] = np.log(spy_price / spy_price.shift(1))

    return spy_data[["Date", "spy_log_return"]]


def build_vix_features(vix_path: Path) -> pd.DataFrame:
    """
    Create volatility regime features from VIX.
    """
    vix_data = load_price_data(vix_path).copy()

    # We again use Adj Close as the main price field so the pipeline is
    # consistent across all assets. For VIX, adjusted and raw close are
    # typically the same, but one consistent rule keeps the code simpler.
    vix_price = vix_data["Adj Close"].astype(float)

    # VIX is the market fear/volatility index and helps capture market regime.
    # Higher values often signal more uncertainty or stress in the market.
    vix_data["vix_close"] = vix_price

    # This return feature captures how quickly market fear is rising or falling.
    vix_data["vix_log_return"] = np.log(vix_price / vix_price.shift(1))

    return vix_data[["Date", "vix_close", "vix_log_return"]]


def engineer_stock_features(
    stock_path: Path, spy_features: pd.DataFrame, vix_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the prediction dataset for one stock and merge market features.
    """
    stock_data = load_price_data(stock_path).copy()
    ticker = stock_path.stem

    # We use Adj Close as the main stock price because it adjusts for splits
    # and dividends, which makes daily return features more reliable.
    price = stock_data["Adj Close"].astype(float)
    volume = stock_data["Volume"].astype(float)

    feature_data = pd.DataFrame()
    feature_data["Date"] = stock_data["Date"]
    feature_data["ticker"] = ticker
    feature_data["adj_close"] = price
    feature_data["volume"] = volume

    # This is the daily log return of Adj Close and is the main price movement feature.
    # It is useful because return prediction is usually more stable than predicting raw price.
    feature_data["log_return"] = np.log(price / price.shift(1))

    # This measures how trading activity changed vs the previous day.
    # Sudden volume jumps can signal stronger interest, news, or unusual trading behavior.
    feature_data["volume_change"] = volume.pct_change()

    # RSI is a momentum indicator that can help detect overbought/oversold conditions.
    # Momentum features can help a model learn whether recent buying or selling pressure matters.
    feature_data["rsi_14"] = RSIIndicator(close=price, window=14).rsi()

    macd_indicator = MACD(close=price, window_slow=26, window_fast=12, window_sign=9)

    # MACD is a trend/momentum indicator based on moving averages.
    # It helps describe whether short-term trend is stronger or weaker than longer-term trend.
    feature_data["macd"] = macd_indicator.macd()

    # This is the MACD signal line.
    # It smooths MACD and can help show whether momentum is strengthening or fading.
    feature_data["macd_signal"] = macd_indicator.macd_signal()

    # This is the gap between MACD and its signal line.
    # The difference can help highlight turning points in momentum.
    feature_data["macd_diff"] = macd_indicator.macd_diff()

    # This is rolling 10-day standard deviation of log returns and represents recent risk/instability.
    # Higher short-term volatility often means less stable price behavior.
    feature_data["volatility_10"] = feature_data["log_return"].rolling(window=10).std()

    # This is tomorrow's log return, created by shifting the stock's log return by -1.
    # We shift by one day so today's features are used to predict tomorrow's move,
    # which avoids leaking future information into the model.
    feature_data["target_next_day_return"] = feature_data["log_return"].shift(-1)

    feature_data = feature_data.merge(spy_features, on="Date", how="left")
    feature_data = feature_data.merge(vix_features, on="Date", how="left")

    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.dropna().reset_index(drop=True)

    if feature_data.empty:
        raise ValueError("No rows remain after dropping NaNs.")

    feature_data["Date"] = feature_data["Date"].dt.strftime("%Y-%m-%d")

    return feature_data[OUTPUT_COLUMNS]


def main() -> None:
    """
    Run the full feature engineering pipeline and save the outputs.
    """
    if not RAW_DATA_DIR.exists():
        raise SystemExit(f"Raw data directory not found: {RAW_DATA_DIR}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    spy_path = RAW_DATA_DIR / SPY_FILE
    vix_path = RAW_DATA_DIR / VIX_FILE

    if not spy_path.exists():
        raise SystemExit(f"Required market file not found: {spy_path}")
    if not vix_path.exists():
        raise SystemExit(f"Required market file not found: {vix_path}")

    spy_features = build_spy_features(spy_path)
    vix_features = build_vix_features(vix_path)

    stock_files = sorted(
        csv_path
        for csv_path in RAW_DATA_DIR.glob("*.csv")
        if csv_path.name not in {SPY_FILE, VIX_FILE}
    )

    if not stock_files:
        raise SystemExit(f"No stock CSV files found in {RAW_DATA_DIR}")

    successful_datasets = []
    failed_tickers = []

    for stock_path in stock_files:
        ticker = stock_path.stem

        try:
            rows_before = len(load_price_data(stock_path))
            processed_data = engineer_stock_features(stock_path, spy_features, vix_features)
            rows_after = len(processed_data)

            output_path = PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
            processed_data.to_csv(output_path, index=False)

            successful_datasets.append(processed_data)

            print(f"\nTicker: {ticker}")
            print(f"Rows before feature engineering: {rows_before}")
            print(f"Rows after dropping NaNs: {rows_after}")
            print(f"Final columns created: {', '.join(processed_data.columns)}")
        except Exception as error:
            failed_tickers.append((ticker, str(error)))
            print(f"\nWarning: Failed to process {ticker}: {error}")

    if successful_datasets:
        final_dataset = pd.concat(successful_datasets, ignore_index=True)
        final_dataset = final_dataset.sort_values(["Date", "ticker"]).reset_index(drop=True)
    else:
        final_dataset = pd.DataFrame(columns=OUTPUT_COLUMNS)

    final_dataset.to_csv(FINAL_DATASET_PATH, index=False)

    print("\nFeature engineering summary")
    print(f"Successful stocks: {len(successful_datasets)}")
    print(f"Failed stocks: {len(failed_tickers)}")
    print(f"Combined dataset rows: {len(final_dataset)}")
    print(f"Combined dataset saved to: {FINAL_DATASET_PATH}")

    if failed_tickers:
        print("Failed ticker details:")
        for ticker, error_message in failed_tickers:
            print(f"  - {ticker}: {error_message}")


if __name__ == "__main__":
    main()
