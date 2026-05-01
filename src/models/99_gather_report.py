"""
Gather model output artifacts into performance report tables.

Reads:
    outputs/**/metrics_1d.json
    outputs/**/metrics_5d.json
    outputs/**/per_ticker_1d.csv
    outputs/**/per_ticker_5d.csv

Writes:
    outputs/reports/performance_report.md
    outputs/reports/model_performance_1d.csv
    outputs/reports/model_performance_5d.csv
    outputs/reports/stock_model_performance_1d.csv
    outputs/reports/stock_model_performance_5d.csv

Run:
    python src/models/99_gather_report.py
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUTS_DIR / "reports"

HORIZONS = {
    "1d": "1 Day",
    "5d": "5 Day",
}

SPLITS = ("train", "validation", "test")
STAT_METRICS = ("mae", "rmse", "directional_accuracy")
BACKTEST_BOOKS = ("long_only", "long_short", "buy_and_hold_benchmark")
BACKTEST_METRICS = (
    "final_value",
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "n_trading_days",
)

MODEL_CSV_COLUMNS = [
    "horizon",
    "model",
    "target",
    "model_dir",
    "train_mae",
    "train_rmse",
    "train_directional_accuracy",
    "validation_mae",
    "validation_rmse",
    "validation_directional_accuracy",
    "test_mae",
    "test_rmse",
    "test_directional_accuracy",
    "long_only_final_value",
    "long_only_total_return",
    "long_only_sharpe_ratio",
    "long_only_max_drawdown",
    "long_only_n_trading_days",
    "long_short_final_value",
    "long_short_total_return",
    "long_short_sharpe_ratio",
    "long_short_max_drawdown",
    "long_short_n_trading_days",
    "buy_and_hold_benchmark_final_value",
    "buy_and_hold_benchmark_total_return",
    "buy_and_hold_benchmark_sharpe_ratio",
    "buy_and_hold_benchmark_max_drawdown",
    "buy_and_hold_benchmark_n_trading_days",
    "backtest_error",
]

STOCK_CSV_COLUMNS = [
    "horizon",
    "ticker",
    "model",
    "mae",
    "rmse",
    "directional_accuracy",
    "model_dir",
]

MODEL_REPORT_COLUMNS = [
    ("model", "Model"),
    ("test_mae", "Test MAE"),
    ("test_rmse", "Test RMSE"),
    ("test_directional_accuracy", "Dir Acc"),
    ("long_only_total_return", "LO Return"),
    ("long_only_sharpe_ratio", "LO Sharpe"),
    ("long_only_max_drawdown", "LO Max DD"),
    ("long_short_total_return", "LS Return"),
    ("long_short_sharpe_ratio", "LS Sharpe"),
    ("long_short_max_drawdown", "LS Max DD"),
    ("buy_and_hold_benchmark_total_return", "BH Return"),
]

STOCK_REPORT_COLUMNS = [
    ("ticker", "Ticker"),
    ("model", "Model"),
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("directional_accuracy", "Dir Acc"),
]


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON with a filename-rich error message."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON: {path}") from exc


def relative_path(path: Path) -> str:
    """Return a stable project-relative path when possible."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def model_name_from_dir(model_dir: Path, metrics: dict[str, Any] | None = None) -> str:
    """Infer a readable model name from metrics JSON or the outputs path."""
    metrics = metrics or {}
    if metrics.get("model"):
        return str(metrics["model"])

    try:
        parts = model_dir.relative_to(OUTPUTS_DIR).parts
    except ValueError:
        return model_dir.name

    if len(parts) >= 2 and parts[0] == "baselines":
        return parts[1]
    if len(parts) >= 2:
        return "_".join(parts[:2])
    if parts:
        return parts[0]
    return model_dir.name


def is_report_artifact(path: Path) -> bool:
    """Avoid re-reading any future report files."""
    return REPORT_DIR in path.parents or path == REPORT_DIR


def add_split_metrics(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Flatten train, validation, and test metric blocks into one row."""
    for split in SPLITS:
        split_metrics = metrics.get(split, {})
        for metric in STAT_METRICS:
            row[f"{split}_{metric}"] = split_metrics.get(metric)


def add_backtest_metrics(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Flatten test backtest blocks into one row."""
    backtest = metrics.get("test_backtest", {})
    row["backtest_error"] = backtest.get("error")

    for book in BACKTEST_BOOKS:
        book_metrics = backtest.get(book, {})
        for metric in BACKTEST_METRICS:
            row[f"{book}_{metric}"] = book_metrics.get(metric)


def collect_model_performance() -> pd.DataFrame:
    """Collect one row per model per horizon from metrics JSON files."""
    rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        pattern = f"metrics_{horizon}.json"
        for path in sorted(OUTPUTS_DIR.rglob(pattern)):
            if is_report_artifact(path):
                continue

            metrics = read_json(path)
            model_dir = path.parent
            row: dict[str, Any] = {
                "horizon": horizon,
                "model": model_name_from_dir(model_dir, metrics),
                "target": metrics.get("target"),
                "model_dir": relative_path(model_dir),
            }
            add_split_metrics(row, metrics)
            add_backtest_metrics(row, metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    return order_columns(df, MODEL_CSV_COLUMNS)


def collect_stock_model_performance() -> pd.DataFrame:
    """Collect one row per ticker, model, and horizon from per-ticker CSVs."""
    frames: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        pattern = f"per_ticker_{horizon}.csv"
        for path in sorted(OUTPUTS_DIR.rglob(pattern)):
            if is_report_artifact(path):
                continue

            metrics_path = path.parent / f"metrics_{horizon}.json"
            metrics = read_json(metrics_path) if metrics_path.exists() else {}
            model_dir = path.parent
            df = pd.read_csv(path)

            if "ticker" not in df.columns:
                raise ValueError(f"Missing ticker column in {path}")

            df = df.copy()
            df.insert(0, "horizon", horizon)
            df.insert(1, "model", model_name_from_dir(model_dir, metrics))
            df["model_dir"] = relative_path(model_dir)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=STOCK_CSV_COLUMNS)

    return order_columns(pd.concat(frames, ignore_index=True), STOCK_CSV_COLUMNS)


def order_columns(df: pd.DataFrame, preferred_columns: list[str]) -> pd.DataFrame:
    """Put expected columns first without dropping unexpected useful columns."""
    if df.empty:
        return pd.DataFrame(columns=preferred_columns)

    first = [column for column in preferred_columns if column in df.columns]
    rest = [column for column in df.columns if column not in first]
    return df[first + rest]


def sort_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sort aggregate rows by test RMSE, then model name."""
    if df.empty:
        return df

    sort_columns = [column for column in ("test_rmse", "test_mae", "model") if column in df.columns]
    return df.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def sort_stock_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sort stock rows by ticker and best RMSE within each ticker."""
    if df.empty:
        return df

    sort_columns = [column for column in ("ticker", "rmse", "mae", "model") if column in df.columns]
    return df.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def write_csv_tables(model_df: pd.DataFrame, stock_df: pd.DataFrame) -> list[Path]:
    """Write separate aggregate and stock-model CSV tables for each horizon."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for horizon in HORIZONS:
        model_path = REPORT_DIR / f"model_performance_{horizon}.csv"
        model_table = sort_model_table(model_df[model_df["horizon"] == horizon]) if not model_df.empty else model_df
        model_table.to_csv(model_path, index=False)
        written.append(model_path)

        stock_path = REPORT_DIR / f"stock_model_performance_{horizon}.csv"
        stock_table = sort_stock_table(stock_df[stock_df["horizon"] == horizon]) if not stock_df.empty else stock_df
        stock_table.to_csv(stock_path, index=False)
        written.append(stock_path)

    return written


def format_cell(column: str, value: Any) -> str:
    """Format a value for Markdown report tables."""
    if pd.isna(value):
        return ""

    if column == "model" or column == "ticker":
        return str(value)
    if column.endswith("directional_accuracy"):
        return f"{float(value):.2%}"
    if column.endswith("total_return") or column.endswith("max_drawdown"):
        return f"{float(value):.2%}"
    if column.endswith("final_value"):
        return f"${float(value):,.2f}"
    if column.endswith("sharpe_ratio"):
        return f"{float(value):.3f}"
    if column in {"mae", "rmse"} or column.endswith("_mae") or column.endswith("_rmse"):
        return f"{float(value):.6f}"

    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    """Render a compact Markdown table without optional dependencies."""
    if df.empty:
        return "_No rows found._"

    present_columns = [(column, label) for column, label in columns if column in df.columns]
    headers = [label for _, label in present_columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = [format_cell(column, row[column]).replace("|", "\\|") for column, _ in present_columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def report_lines(model_df: pd.DataFrame, stock_df: pd.DataFrame, csv_paths: list[Path]) -> list[str]:
    """Build the Markdown report body."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Model Performance Report",
        "",
        f"Generated: {generated_at}",
        f"Source directory: {relative_path(OUTPUTS_DIR)}",
        "",
    ]

    for horizon, label in HORIZONS.items():
        model_table = model_df[model_df["horizon"] == horizon] if not model_df.empty else model_df
        model_table = sort_model_table(model_table)
        lines.extend(
            [
                f"## Aggregate Model Performance - {label}",
                "",
                markdown_table(model_table, MODEL_REPORT_COLUMNS),
                "",
            ]
        )

    for horizon, label in HORIZONS.items():
        stock_table = stock_df[stock_df["horizon"] == horizon] if not stock_df.empty else stock_df
        stock_table = sort_stock_table(stock_table)
        lines.extend(
            [
                f"## Stock x Model Performance - {label}",
                "",
                markdown_table(stock_table, STOCK_REPORT_COLUMNS),
                "",
            ]
        )

    lines.extend(
        [
            "## CSV Outputs",
            "",
            *[f"- {relative_path(path)}" for path in csv_paths],
            "",
        ]
    )

    if model_df.empty and stock_df.empty:
        lines.extend(
            [
                "No model output artifacts were found yet. Run the training scripts first, then rerun this report.",
                "",
            ]
        )

    return lines


def write_markdown_report(model_df: pd.DataFrame, stock_df: pd.DataFrame, csv_paths: list[Path]) -> Path:
    """Write the human-readable Markdown report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "performance_report.md"
    report_path.write_text("\n".join(report_lines(model_df, stock_df, csv_paths)), encoding="utf-8")
    return report_path


def main() -> None:
    """Gather all outputs and write report artifacts."""
    model_df = collect_model_performance()
    stock_df = collect_stock_model_performance()

    csv_paths = write_csv_tables(model_df, stock_df)
    report_path = write_markdown_report(model_df, stock_df, csv_paths)

    model_count = model_df["model"].nunique() if not model_df.empty else 0
    stock_count = stock_df["ticker"].nunique() if not stock_df.empty else 0

    print("Report generated successfully.")
    print(f"  Models found : {model_count}")
    print(f"  Stocks found : {stock_count}")
    print(f"  Report       : {report_path}")
    print(f"  CSV tables   : {REPORT_DIR}")


if __name__ == "__main__":
    main()
