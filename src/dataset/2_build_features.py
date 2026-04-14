"""
Build a clean feature engineering dataset for daily stock return prediction.

Reads raw stock CSV files from data/raw/, computes technical and market
features, creates a next-day return target, and saves both per-stock
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
        "Missing dependency 'ta'. Install with: pip install pandas numpy ta"
    ) from error


# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DATA_DIR       = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
FINAL_DATASET_PATH = Path("data/final_model_dataset.csv")

SPY_FILE = "SPY.csv"
VIX_FILE = "VIX.csv"   # saved as VIX.csv by download_yahoo_data.py (^ stripped)

# ── Output schema ──────────────────────────────────────────────────────────────

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


# ── Data loading ───────────────────────────────────────────────────────────────

def load_price_data(csv_path: Path) -> pd.DataFrame:
    """
    Read one raw CSV file, parse Date as datetime, and sort ascending by Date.
    Raises ValueError if required columns are missing or dates are invalid.
    """
    df = pd.read_csv(csv_path)

    required = {"Date", "Adj Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"[{csv_path.name}] Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError(f"[{csv_path.name}] Date column contains unparseable values.")

    return df.sort_values("Date").reset_index(drop=True)


# ── Market feature builders ────────────────────────────────────────────────────

def build_spy_features(spy_path: Path) -> pd.DataFrame:
    """
    Compute daily log return from SPY Adj Close.
    Used as a market direction proxy — tells the model whether the broad
    market moved up or down on a given day.
    Returns a DataFrame with columns [Date, spy_log_return].
    """
    df = load_price_data(spy_path)
    price = df["Adj Close"].astype(float)
    df["spy_log_return"] = np.log(price / price.shift(1))
    return df[["Date", "spy_log_return"]]


def build_vix_features(vix_path: Path) -> pd.DataFrame:
    """
    Extract VIX level and its daily log return.
    VIX level captures the current fear/volatility regime.
    VIX log return captures how quickly fear is rising or falling.
    Returns a DataFrame with columns [Date, vix_close, vix_log_return].
    """
    df = load_price_data(vix_path)
    price = df["Adj Close"].astype(float)
    df["vix_close"]      = price
    df["vix_log_return"] = np.log(price / price.shift(1))
    return df[["Date", "vix_close", "vix_log_return"]]


# ── Per-stock feature engineering ─────────────────────────────────────────────

def engineer_stock_features(
    stock_path: Path,
    spy_features: pd.DataFrame,
    vix_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full feature set for one stock and merge market context.

    Features created:
        log_return      — daily price movement signal
        volume_change   — trading activity change vs previous day
        rsi_14          — 14-day momentum / overbought-oversold indicator
        macd            — trend strength from moving average crossover
        macd_signal     — smoothed MACD line
        macd_diff       — gap between MACD and signal (momentum turning point)
        volatility_10   — 10-day rolling std of log returns (recent risk)
        spy_log_return  — broad market direction (merged from SPY)
        vix_close       — current fear/volatility regime (merged from VIX)
        vix_log_return  — rate of change in market fear (merged from VIX)

    Target:
        target_next_day_return — next day's log return, created by shifting
                                 log_return forward by one day. Today's
                                 features predict tomorrow's return.

    Rows with NaNs (from rolling windows and the target shift) are dropped.
    Returns a DataFrame with columns matching OUTPUT_COLUMNS.
    """
    df     = load_price_data(stock_path)
    ticker = stock_path.stem

    price  = df["Adj Close"].astype(float)
    volume = df["Volume"].astype(float)

    features = pd.DataFrame()
    features["Date"]   = df["Date"]
    features["ticker"] = ticker
    features["adj_close"] = price
    features["volume"]    = volume

    # Price movement
    features["log_return"] = np.log(price / price.shift(1))

    # Trading activity
    features["volume_change"] = volume.pct_change()

    # Momentum indicators
    features["rsi_14"] = RSIIndicator(close=price, window=14).rsi()

    macd_ind = MACD(close=price, window_slow=26, window_fast=12, window_sign=9)
    features["macd"]        = macd_ind.macd()
    features["macd_signal"] = macd_ind.macd_signal()
    features["macd_diff"]   = macd_ind.macd_diff()

    # Recent volatility (risk)
    features["volatility_10"] = features["log_return"].rolling(window=10).std()

    # Target: next-day return (shift by -1 so today's row predicts tomorrow)
    features["target_next_day_return"] = features["log_return"].shift(-1)

    # Merge market-level features by date
    features = features.merge(spy_features, on="Date", how="left")
    features = features.merge(vix_features, on="Date", how="left")

    # Drop rows with any NaN (rolling window warmup + target shift)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna().reset_index(drop=True)

    if features.empty:
        raise ValueError(f"[{ticker}] No rows remain after dropping NaNs.")

    features["Date"] = features["Date"].dt.strftime("%Y-%m-%d")

    return features[OUTPUT_COLUMNS]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run the full feature engineering pipeline for all stocks.

    Steps:
        1. Load SPY and VIX market features once.
        2. Process each stock file in data/raw/ individually.
        3. Save one processed CSV per stock to data/processed/.
        4. Concatenate all stocks into data/final_model_dataset.csv.
    """
    if not RAW_DATA_DIR.exists():
        raise SystemExit(f"Raw data directory not found: {RAW_DATA_DIR}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    spy_path = RAW_DATA_DIR / SPY_FILE
    vix_path = RAW_DATA_DIR / VIX_FILE

    for path in (spy_path, vix_path):
        if not path.exists():
            raise SystemExit(f"Required market file not found: {path}")

    spy_features = build_spy_features(spy_path)
    vix_features = build_vix_features(vix_path)

    stock_files = sorted(
        p for p in RAW_DATA_DIR.glob("*.csv")
        if p.name not in {SPY_FILE, VIX_FILE}
    )

    if not stock_files:
        raise SystemExit(f"No stock CSV files found in {RAW_DATA_DIR}")

    successful: list[pd.DataFrame] = []
    failed:     list[tuple[str, str]] = []

    for stock_path in stock_files:
        ticker = stock_path.stem
        try:
            rows_raw = len(load_price_data(stock_path))
            processed = engineer_stock_features(stock_path, spy_features, vix_features)
            rows_processed = len(processed)

            out_path = PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
            processed.to_csv(out_path, index=False)
            successful.append(processed)

            print(f"  OK  {ticker:<6}  raw={rows_raw}  processed={rows_processed}")
        except Exception as error:
            failed.append((ticker, str(error)))
            print(f"  FAIL {ticker:<6}  {error}")

    # Combine all processed stocks into one final dataset
    if successful:
        final = pd.concat(successful, ignore_index=True)
        final = final.sort_values(["Date", "ticker"]).reset_index(drop=True)
    else:
        final = pd.DataFrame(columns=OUTPUT_COLUMNS)

    final.to_csv(FINAL_DATASET_PATH, index=False)

    print("\n── Feature engineering summary ───────────────────────")
    print(f"Successful : {len(successful)} / {len(stock_files)}")
    print(f"Failed     : {len(failed)}")
    print(f"Total rows : {len(final)}")
    print(f"Saved to   : {FINAL_DATASET_PATH}")

    if failed:
        print("\nFailed ticker details:")
        for ticker, error_message in failed:
            print(f"  {ticker:<6} {error_message}")


if __name__ == "__main__":
    main()