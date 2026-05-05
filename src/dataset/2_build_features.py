"""
Build a clean feature engineering dataset for daily stock return prediction.

Reads raw stock CSV files from data/raw/, computes technical, market, sector,
and risk features, creates 1/5/10/30-day return targets, and saves both per-stock
processed files and one combined modeling dataset.

Features:
    Price:      log_return, return_5d, return_10d, return_30d
    Volume:     volume_change, volume_ma_ratio, obv_change
    Momentum:   rsi_14, macd, macd_signal, macd_diff, rolling_sharpe_20
    Volatility: volatility_10, volatility_20, volatility_30, atr_14,
                bollinger_band_width, beta_60, idiosyncratic_vol_20,
                volatility_regime_20
    Market:     SPY/QQQ/DIA/IWM returns, VIX, relative strength
    Sector:     sector momentum and stock-vs-sector strength
    Calendar:   day_of_week

Targets:
    target_next_day_return  — next trading day's log return
    target_5d_return        — cumulative 5-day forward log return
    target_10d_return       — cumulative 10-day forward log return
    target_30d_return       — cumulative 30-day forward log return

Install dependencies with:
    pip install pandas numpy ta
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from universes import get_data_dir, SECTOR_MAP

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands, AverageTrueRange
except ImportError as error:
    raise SystemExit(
        "Missing dependency 'ta'. Install with: pip install pandas numpy ta"
    ) from error


# ── Paths ──────────────────────────────────────────────────────────────────────

_DATA              = get_data_dir()
RAW_DATA_DIR       = _DATA / "raw"
PROCESSED_DATA_DIR = _DATA / "processed"
FINAL_DATASET_PATH = _DATA / "final_model_dataset.csv"

SPY_FILE = "SPY.csv"
VIX_FILE = "VIX.csv"
MARKET_INDEX_FILES = {
    "qqq": "QQQ.csv",
    "dia": "DIA.csv",
    "iwm": "IWM.csv",
}

REFERENCE_COLUMNS = [
    "Date",
    "ticker",
    "adj_close",
    "volume",
]

TARGET_COLUMNS = [
    "target_next_day_return",
    "target_5d_return",
    "target_10d_return",
    "target_30d_return",
]

FEATURE_COLUMNS = [
    # Price features
    "log_return",
    "return_5d",
    "return_10d",
    "return_30d",
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
    # Volatility and risk features
    "volatility_10",
    "volatility_20",
    "volatility_30",
    "atr_14",
    "bollinger_band_width",
    "beta_60",
    "idiosyncratic_vol_20",
    "volatility_regime_20",
    # Market context features
    "spy_log_return",
    "spy_return_5d",
    "spy_return_10d",
    "spy_return_30d",
    "spy_volatility_20",
    "qqq_log_return",
    "qqq_return_5d",
    "dia_log_return",
    "dia_return_5d",
    "iwm_log_return",
    "iwm_return_5d",
    "vix_close",
    "vix_log_return",
    "relative_strength",
    # Sector features
    "sector_log_return",
    "sector_return_5d",
    "sector_return_10d",
    "sector_return_30d",
    "sector_relative_strength",
    # Calendar features
    "day_of_week",
]
SECTOR_FEATURE_COLUMNS = [
    "sector_log_return",
    "sector_return_5d",
    "sector_return_10d",
    "sector_return_30d",
    "sector_relative_strength",
]
BASE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in SECTOR_FEATURE_COLUMNS]

# SECTOR_MAP imported from universes.py (covers both tech30 and energy30)

# ── Output schema ──────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = REFERENCE_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


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
    """Compute SPY market direction, trend, and volatility features."""
    df = load_price_data(spy_path)
    price = df["Adj Close"].astype(float)
    df["spy_log_return"] = np.log(price / price.shift(1))
    df["spy_return_5d"] = df["spy_log_return"].rolling(window=5).sum()
    df["spy_return_10d"] = df["spy_log_return"].rolling(window=10).sum()
    df["spy_return_30d"] = df["spy_log_return"].rolling(window=30).sum()
    df["spy_volatility_20"] = df["spy_log_return"].rolling(window=20).std()
    return df[[
        "Date",
        "spy_log_return",
        "spy_return_5d",
        "spy_return_10d",
        "spy_return_30d",
        "spy_volatility_20",
    ]]


def build_index_features(raw_dir: Path, name: str, filename: str, dates: pd.Series) -> pd.DataFrame:
    """Build optional ETF context features, using neutral zeros if data is absent."""
    columns = ["Date", f"{name}_log_return", f"{name}_return_5d"]
    path = raw_dir / filename
    if not path.exists():
        out = pd.DataFrame({"Date": pd.to_datetime(dates).drop_duplicates().sort_values()})
        out[f"{name}_log_return"] = 0.0
        out[f"{name}_return_5d"] = 0.0
        print(f"  WARN missing {filename}; {name.upper()} features filled with 0")
        return out[columns]

    df = load_price_data(path)
    price = df["Adj Close"].astype(float)
    df[f"{name}_log_return"] = np.log(price / price.shift(1))
    df[f"{name}_return_5d"] = df[f"{name}_log_return"].rolling(window=5).sum()
    return df[columns]


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
    index_features: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the full feature set for one stock and merge market context.

    Input features cover price, volume, momentum, volatility, market,
    sector, and calendar effects. Four targets are produced:
    1-day, 5-day, 10-day, and 30-day forward cumulative returns.
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
    features["return_30d"] = features["log_return"].rolling(window=30).sum()

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
    features["volatility_20"] = features["log_return"].rolling(window=20).std()
    features["volatility_30"] = features["log_return"].rolling(window=30).std()

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
    features["target_10d_return"] = features["log_return"].rolling(window=10).sum().shift(-10)
    features["target_30d_return"] = features["log_return"].rolling(window=30).sum().shift(-30)

    # ── Merge market features ──────────────────────────────────────────────────

    features = features.merge(spy_features, on="Date", how="left")
    features = features.merge(vix_features, on="Date", how="left")
    for index_df in index_features:
        features = features.merge(index_df, on="Date", how="left")

    # ── Cross-market features ──────────────────────────────────────────────────

    # Relative strength: stock return minus market return
    # Captures whether the stock is outperforming or underperforming SPY
    features["relative_strength"] = features["log_return"] - features["spy_log_return"]

    # Beta and residual volatility: captures market sensitivity and stock-specific risk.
    rolling_cov = features["log_return"].rolling(window=60).cov(features["spy_log_return"])
    rolling_var = features["spy_log_return"].rolling(window=60).var()
    features["beta_60"] = rolling_cov / rolling_var
    residual_return = features["log_return"] - features["beta_60"] * features["spy_log_return"]
    features["idiosyncratic_vol_20"] = residual_return.rolling(window=20).std()
    long_run_vol = features["volatility_20"].rolling(window=252, min_periods=60).median()
    features["volatility_regime_20"] = features["volatility_20"] / long_run_vol

    # ── Calendar features ──────────────────────────────────────────────────────

    # Day of week: 0=Monday, 4=Friday. Captures weekday effects.
    features["day_of_week"] = features["Date"].dt.dayofweek

    # ── Clean up ───────────────────────────────────────────────────────────────

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna().reset_index(drop=True)

    if features.empty:
        raise ValueError(f"[{ticker}] No rows remain after dropping NaNs.")

    features["Date"] = features["Date"].dt.strftime("%Y-%m-%d")

    return features[REFERENCE_COLUMNS + BASE_FEATURE_COLUMNS + TARGET_COLUMNS]


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leave-one-out sector momentum and relative-strength features."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["sector"] = out["ticker"].map(SECTOR_MAP).fillna("other")

    grouped = out.groupby(["Date", "sector"])["log_return"]
    sector_sum = grouped.transform("sum")
    sector_count = grouped.transform("count")
    sector_mean = grouped.transform("mean")
    leave_one_out = (sector_sum - out["log_return"]) / (sector_count - 1)
    out["sector_log_return"] = np.where(sector_count > 1, leave_one_out, sector_mean)

    out = out.sort_values(["ticker", "Date"]).reset_index(drop=True)
    for window in (5, 10, 30):
        out[f"sector_return_{window}d"] = (
            out.groupby("ticker")["sector_log_return"]
            .transform(lambda s: s.rolling(window=window).sum())
        )
    out["sector_relative_strength"] = out["log_return"] - out["sector_log_return"]

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=OUTPUT_COLUMNS).reset_index(drop=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out[OUTPUT_COLUMNS]


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
    index_features = [
        build_index_features(RAW_DATA_DIR, name, filename, spy_features["Date"])
        for name, filename in MARKET_INDEX_FILES.items()
    ]

    stock_files = sorted(
        p for p in RAW_DATA_DIR.glob("*.csv")
        if p.name not in {SPY_FILE, VIX_FILE, *MARKET_INDEX_FILES.values()}
    )

    if not stock_files:
        raise SystemExit(f"No stock CSV files found in {RAW_DATA_DIR}")

    successful: list[pd.DataFrame] = []
    failed:     list[tuple[str, str]] = []

    for stock_path in stock_files:
        ticker = stock_path.stem
        try:
            rows_raw = len(load_price_data(stock_path))
            processed = engineer_stock_features(stock_path, spy_features, vix_features, index_features)
            rows_processed = len(processed)
            successful.append(processed)

            print(f"  OK  {ticker:<6}  raw={rows_raw}  processed={rows_processed}")
        except Exception as error:
            failed.append((ticker, str(error)))
            print(f"  FAIL {ticker:<6}  {error}")

    if successful:
        final = pd.concat(successful, ignore_index=True)
        final = add_sector_features(final)
        final = final.sort_values(["Date", "ticker"]).reset_index(drop=True)
        for ticker, processed in final.groupby("ticker", sort=True):
            out_path = PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
            processed.to_csv(out_path, index=False)
    else:
        final = pd.DataFrame(columns=OUTPUT_COLUMNS)

    final.to_csv(FINAL_DATASET_PATH, index=False)

    print("\n── Feature engineering summary ───────────────────────")
    print(f"Successful : {len(successful)} / {len(stock_files)}")
    print(f"Failed     : {len(failed)}")
    print(f"Total rows : {len(final)}")
    print(f"Features   : {len(FEATURE_COLUMNS)} input + {len(TARGET_COLUMNS)} targets")
    print(f"Saved to   : {FINAL_DATASET_PATH}")

    if failed:
        print("\nFailed ticker details:")
        for ticker, error_message in failed:
            print(f"  {ticker:<6} {error_message}")


if __name__ == "__main__":
    main()
