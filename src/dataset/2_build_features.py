"""
Build a clean feature engineering dataset for daily stock return prediction.

Reads raw stock CSV files from data/raw/, computes technical and market
features, creates next-day and 5-day return targets, and saves both per-stock
processed files and one combined modeling dataset.

Features (19 total):
    Price:      log_return, return_5d, return_10d
    Volume:     volume_change, volume_ma_ratio, obv_change
    Momentum:   rsi_14, macd, macd_signal, macd_diff, rolling_sharpe_20
    Volatility: volatility_10, atr_14, bollinger_band_width
    Market:     spy_log_return, vix_close, vix_log_return, relative_strength
    Calendar:   day_of_week

Targets:
    target_next_day_return  — next trading day's log return
    target_5d_return        — cumulative 5-day forward log return

Install dependencies with:
    pip install pandas numpy ta
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands, AverageTrueRange
except ImportError as error:
    raise SystemExit(
        "Missing dependency 'ta'. Install with: pip install pandas numpy ta"
    ) from error


# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DATA_DIR       = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
FINAL_DATASET_PATH = Path("data/final_model_dataset.csv")

SPY_FILE = "SPY.csv"
VIX_FILE = "VIX.csv"

# ── Output schema ──────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "Date",
    "ticker",
    "adj_close",
    "volume",
    # Price features
    "log_return",
    "return_5d",
    "return_10d",
    # Volume features
    "volume_change",
    "volume_ma_ratio",
    "obv_change",
    # Momentum features
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "rolling_sharpe_20",
    # Volatility features
    "volatility_10",
    "atr_14",
    "bollinger_band_width",
    # Market context features
    "spy_log_return",
    "vix_close",
    "vix_log_return",
    "relative_strength",
    # Calendar features
    "day_of_week",
    # Targets
    "target_next_day_return",
    "target_5d_return",
]


def load_price_data(csv_path: Path) -> pd.DataFrame:
    """Read raw CSV, parse dates, sort ascending. Validates required columns."""
    df = pd.read_csv(csv_path)

    required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"[{csv_path.name}] Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError(f"[{csv_path.name}] Date column contains unparseable values.")

    return df.sort_values("Date").reset_index(drop=True)


def build_spy_features(spy_path: Path) -> pd.DataFrame:
    """Compute daily log return from SPY Adj Close. Used for market direction."""
    df = load_price_data(spy_path)
    price = df["Adj Close"].astype(float)
    df["spy_log_return"] = np.log(price / price.shift(1))
    return df[["Date", "spy_log_return"]]


def build_vix_features(vix_path: Path) -> pd.DataFrame:
    """Extract VIX level and its daily log return — captures fear regime."""
    df = load_price_data(vix_path)
    price = df["Adj Close"].astype(float)
    df["vix_close"]      = price
    df["vix_log_return"] = np.log(price / price.shift(1))
    return df[["Date", "vix_close", "vix_log_return"]]


def engineer_stock_features(
    stock_path: Path,
    spy_features: pd.DataFrame,
    vix_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full feature set for one stock and merge market context.

    19 input features cover price, volume, momentum, volatility,
    market context, and calendar effects. Two targets are produced:
    next-day return and 5-day forward cumulative return.
    """
    df     = load_price_data(stock_path)
    ticker = stock_path.stem

    price  = df["Adj Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    features = pd.DataFrame()
    features["Date"]      = df["Date"]
    features["ticker"]    = ticker
    features["adj_close"] = price
    features["volume"]    = volume

    # ── Price features ─────────────────────────────────────────────────────────

    features["log_return"] = np.log(price / price.shift(1))
    features["return_5d"]  = features["log_return"].rolling(window=5).sum()
    features["return_10d"] = features["log_return"].rolling(window=10).sum()

    # ── Volume features ────────────────────────────────────────────────────────

    raw_volume_change = volume.pct_change()
    lower = raw_volume_change.quantile(0.01)
    upper = raw_volume_change.quantile(0.99)
    features["volume_change"] = raw_volume_change.clip(lower=lower, upper=upper)

    volume_ma_20 = volume.rolling(window=20).mean()
    features["volume_ma_ratio"] = volume / volume_ma_20

    obv_direction = np.where(price > price.shift(1), volume,
                    np.where(price < price.shift(1), -volume, 0))
    obv = pd.Series(obv_direction, index=df.index).cumsum()
    features["obv_change"] = obv.pct_change()

    obv_lower = features["obv_change"].quantile(0.01)
    obv_upper = features["obv_change"].quantile(0.99)
    features["obv_change"] = features["obv_change"].clip(lower=obv_lower, upper=obv_upper)

    # ── Momentum features ──────────────────────────────────────────────────────

    features["rsi_14"] = RSIIndicator(close=price, window=14).rsi()

    macd_ind = MACD(close=price, window_slow=26, window_fast=12, window_sign=9)
    features["macd"]        = macd_ind.macd()
    features["macd_signal"] = macd_ind.macd_signal()
    features["macd_diff"]   = macd_ind.macd_diff()

    # 20-day rolling Sharpe ratio — risk-adjusted momentum signal
    rolling_mean = features["log_return"].rolling(window=20).mean()
    rolling_std  = features["log_return"].rolling(window=20).std()
    features["rolling_sharpe_20"] = rolling_mean / rolling_std

    # ── Volatility features ────────────────────────────────────────────────────

    features["volatility_10"] = features["log_return"].rolling(window=10).std()

    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    features["atr_14"] = atr.average_true_range()

    bb = BollingerBands(close=price, window=20, window_dev=2)
    bb_upper  = bb.bollinger_hband()
    bb_lower  = bb.bollinger_lband()
    bb_middle = bb.bollinger_mavg()
    features["bollinger_band_width"] = (bb_upper - bb_lower) / bb_middle

    # ── Targets ────────────────────────────────────────────────────────────────

    features["target_next_day_return"] = features["log_return"].shift(-1)
    features["target_5d_return"] = features["log_return"].rolling(window=5).sum().shift(-5)

    # ── Merge market features ──────────────────────────────────────────────────

    features = features.merge(spy_features, on="Date", how="left")
    features = features.merge(vix_features, on="Date", how="left")

    # ── Cross-market features ──────────────────────────────────────────────────

    # Relative strength: stock return minus market return
    # Captures whether the stock is outperforming or underperforming SPY
    features["relative_strength"] = features["log_return"] - features["spy_log_return"]

    # ── Calendar features ──────────────────────────────────────────────────────

    # Day of week: 0=Monday, 4=Friday. Captures weekday effects.
    features["day_of_week"] = features["Date"].dt.dayofweek

    # ── Clean up ───────────────────────────────────────────────────────────────

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna().reset_index(drop=True)

    if features.empty:
        raise ValueError(f"[{ticker}] No rows remain after dropping NaNs.")

    features["Date"] = features["Date"].dt.strftime("%Y-%m-%d")

    return features[OUTPUT_COLUMNS]


def main() -> None:
    """Run the full feature engineering pipeline for all stocks."""
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

    if successful:
        final = pd.concat(successful, ignore_index=True)
        final = final.sort_values(["Date", "ticker"]).reset_index(drop=True)
    else:
        final = pd.DataFrame(columns=OUTPUT_COLUMNS)

    final.to_csv(FINAL_DATASET_PATH, index=False)

    n_features = len(OUTPUT_COLUMNS) - 6  # minus Date, ticker, adj_close, volume, 2 targets

    print("\n── Feature engineering summary ───────────────────────")
    print(f"Successful : {len(successful)} / {len(stock_files)}")
    print(f"Failed     : {len(failed)}")
    print(f"Total rows : {len(final)}")
    print(f"Features   : {n_features} input + 2 targets")
    print(f"Saved to   : {FINAL_DATASET_PATH}")

    if failed:
        print("\nFailed ticker details:")
        for ticker, error_message in failed:
            print(f"  {ticker:<6} {error_message}")


if __name__ == "__main__":
    main()
