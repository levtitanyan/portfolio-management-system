"""
Refresh saved backtest metrics from existing prediction files.

Use this after changing portfolio logic when model predictions already exist
and retraining would be unnecessary.

Run:
    python src/models/98_refresh_backtests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_utils import run_model_backtest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from universes import get_data_dir, get_outputs_dir

BASE_DIR    = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = get_outputs_dir()
TEST_PATH   = get_data_dir() / "splits" / "test.csv"

INITIAL_CAPITAL = 10_000.0
TOP_K_LONGS = 5
BOTTOM_K_SHORTS = 5
TRADING_DAYS_PER_YEAR = 252

HORIZONS = {
    "1d": 1,
    "5d": 5,
    "10d": 10,
    "30d": 30,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_reference() -> pd.DataFrame:
    ref = pd.read_csv(TEST_PATH)
    ref["Date"] = pd.to_datetime(ref["Date"])
    keep = ["Date", "ticker", "adj_close", "target_next_day_return"]
    missing = [column for column in keep if column not in ref.columns]
    if missing:
        raise ValueError(f"Missing reference columns: {missing}")
    return ref[keep].copy()


def enrich_predictions(preds: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    preds = preds.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    enriched = preds.merge(reference, on=["Date", "ticker"], how="left", suffixes=("", "_ref"))

    if "adj_close_ref" in enriched.columns and "adj_close" not in enriched.columns:
        enriched["adj_close"] = enriched["adj_close_ref"]
    if "target_next_day_return_ref" in enriched.columns and "target_next_day_return" not in enriched.columns:
        enriched["target_next_day_return"] = enriched["target_next_day_return_ref"]

    drop_cols = [column for column in enriched.columns if column.endswith("_ref")]
    if drop_cols:
        enriched = enriched.drop(columns=drop_cols)
    return enriched


def refresh_one(metrics_path: Path, horizon: str, reference: pd.DataFrame) -> bool:
    pred_path = metrics_path.parent / f"predictions_{horizon}_test.csv"
    if not pred_path.exists():
        return False

    metrics = read_json(metrics_path)
    preds = pd.read_csv(pred_path)
    required = {"Date", "ticker", "y_true", "y_pred"}
    if not required.issubset(preds.columns):
        missing = sorted(required - set(preds.columns))
        raise ValueError(f"{pred_path} missing columns: {missing}")

    enriched = enrich_predictions(preds, reference)
    backtest = run_model_backtest(
        enriched,
        enriched["y_pred"].to_numpy(),
        "y_true",
        holding_days=HORIZONS[horizon],
        benchmark_df=reference,
        initial_capital=INITIAL_CAPITAL,
        top_k_longs=TOP_K_LONGS,
        bottom_k_shorts=BOTTOM_K_SHORTS,
        trading_days_per_year=TRADING_DAYS_PER_YEAR,
    )

    metrics["test_backtest"] = backtest
    write_json(metrics_path, metrics)
    return True


def main() -> None:
    reference = load_reference()
    refreshed = 0

    for horizon in HORIZONS:
        for metrics_path in sorted(OUTPUTS_DIR.rglob(f"metrics_{horizon}.json")):
            if "reports" in metrics_path.parts or "portfolio" in metrics_path.parts:
                continue
            if refresh_one(metrics_path, horizon, reference):
                refreshed += 1

    print(f"Refreshed backtests: {refreshed}")


if __name__ == "__main__":
    main()
