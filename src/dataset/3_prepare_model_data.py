"""
Split the combined feature dataset into train, validation, and test sets.

Splits are chronological and strictly non-overlapping:
    Train      : up to and including 2021-12-31
    Validation : 2022-01-01 to 2023-12-31
    Test       : 2024-01-01 onward

Run from the project root:
    python src/dataset/3_prepare_model_data.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from universes import get_data_dir

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR           = Path(__file__).resolve().parents[2]
_DATA              = get_data_dir()
INPUT_DATASET_PATH = _DATA / "final_model_dataset.csv"
SPLITS_DIR         = _DATA / "splits"

TRAIN_PATH = SPLITS_DIR / "train.csv"
VAL_PATH   = SPLITS_DIR / "val.csv"
TEST_PATH  = SPLITS_DIR / "test.csv"

# ── Split boundaries ───────────────────────────────────────────────────────────

TRAIN_END_DATE = "2021-12-31"   # Train: 2015–2021 (7 years)
VAL_END_DATE   = "2023-12-31"   # Val:   2022–2023 (2 years), Test: 2024+ (~2 years)

# ── Column definitions ─────────────────────────────────────────────────────────

TARGET_COLUMNS = [
    "target_next_day_return",
    "target_5d_return",
    "target_10d_return",
    "target_30d_return",
]

FEATURE_COLUMNS = [
    # Price
    "log_return",
    "return_5d",
    "return_10d",
    "return_30d",
    # Volume
    "volume_change",
    "volume_ma_ratio",
    "obv_change",
    # Momentum
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "rolling_sharpe_20",
    # Volatility
    "volatility_10",
    "volatility_20",
    "volatility_30",
    "atr_14",
    "bollinger_band_width",
    "beta_60",
    "idiosyncratic_vol_20",
    "volatility_regime_20",
    # Market context
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
    # Sector
    "sector_log_return",
    "sector_return_5d",
    "sector_return_10d",
    "sector_return_30d",
    "sector_relative_strength",
    # Calendar
    "day_of_week",
]

REFERENCE_COLUMNS = [
    "Date",
    "ticker",
    "adj_close",
    "volume",
]

EXPECTED_COLUMNS = REFERENCE_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the combined feature dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Date column contains unparseable values.")

    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    if df[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    return df


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train, validation, and test sets."""
    train_end = pd.Timestamp(TRAIN_END_DATE)
    val_end   = pd.Timestamp(VAL_END_DATE)

    train_df = df[df["Date"] <= train_end].copy()
    val_df   = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()
    test_df  = df[df["Date"] > val_end].copy()

    for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        if split.empty:
            raise ValueError(f"{name} split is empty. Check date boundaries.")

    return train_df, val_df, test_df


def validate_split_order(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Confirm splits are strictly ordered with no overlap."""
    if train_df["Date"].max() >= val_df["Date"].min():
        raise ValueError("Train and validation splits overlap.")
    if val_df["Date"].max() >= test_df["Date"].min():
        raise ValueError("Validation and test splits overlap.")


def save_split(df: pd.DataFrame, output_path: Path) -> None:
    """Save one split to CSV with Date formatted as YYYY-MM-DD."""
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)


def summarize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print a compact summary table of all three splits."""
    print("\n── Split summary ─────────────────────────────────────")
    print(f"{'Split':<12} {'Rows':<8} {'Tickers':<10} {'Date range'}")
    print(f"{'─'*12} {'─'*8} {'─'*10} {'─'*30}")

    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        date_min = df["Date"].min().date()
        date_max = df["Date"].max().date()
        print(
            f"{name:<12} {len(df):<8} {df['ticker'].nunique():<10} "
            f"{date_min} -> {date_max}"
        )

    print(f"\nFeatures : {len(FEATURE_COLUMNS)}")
    print(f"Targets  : {', '.join(TARGET_COLUMNS)}")


def main() -> None:
    """Load, split, validate, save, and summarize."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(INPUT_DATASET_PATH)
    train_df, val_df, test_df = split_dataset(df)
    validate_split_order(train_df, val_df, test_df)

    save_split(train_df, TRAIN_PATH)
    save_split(val_df,   VAL_PATH)
    save_split(test_df,  TEST_PATH)

    print("Dataset split completed successfully.")
    print(f"  Train      -> {TRAIN_PATH}")
    print(f"  Validation -> {VAL_PATH}")
    print(f"  Test       -> {TEST_PATH}")

    summarize_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
