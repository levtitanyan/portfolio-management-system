"""
Train baseline models and evaluate them on both prediction targets,
including portfolio-level backtest metrics (long-only and long-short).

Models:
    1. Historical Mean
    2. Naive Last Return  (1-day target only)
    3. Linear Regression
    4. Random Forest (time-weighted training — recent data weighted higher)

Targets:
    - target_next_day_return  (1-day ahead)
    - target_5d_return        (5-day ahead)

Metrics computed per model per target:
    Statistical:  MAE, RMSE, directional accuracy
    Backtest:     long-only return, long-short return, Sharpe, max drawdown,
                  vs buy-and-hold SPY benchmark

Run from the project root:
    python src/models/4_train_baselines.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parents[2]
TRAIN_PATH      = BASE_DIR / "data" / "splits" / "train.csv"
VAL_PATH        = BASE_DIR / "data" / "splits" / "val.csv"
TEST_PATH       = BASE_DIR / "data" / "splits" / "test.csv"
METRICS_DIR     = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"

# ── Backtest configuration ─────────────────────────────────────────────────────

INITIAL_CAPITAL = 10_000.0  # starting amount in dollars
TOP_K_LONGS     = 5         # buy top 5 predicted stocks
BOTTOM_K_SHORTS = 5         # short bottom 5 predicted stocks
TRADING_DAYS_PER_YEAR = 252

# ── Columns ────────────────────────────────────────────────────────────────────

TARGET_COLUMNS = [
    "target_next_day_return",
    "target_5d_return",
]

FEATURE_COLUMNS = [
    "log_return",
    "return_5d",
    "return_10d",
    "volume_change",
    "volume_ma_ratio",
    "obv_change",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "rolling_sharpe_20",
    "volatility_10",
    "atr_14",
    "bollinger_band_width",
    "spy_log_return",
    "vix_close",
    "vix_log_return",
    "relative_strength",
    "day_of_week",
]

REFERENCE_COLUMNS = ["Date", "ticker"]


# ── Loading ────────────────────────────────────────────────────────────────────

def load_split(csv_path: Path) -> pd.DataFrame:
    """Load one split CSV, validate columns, parse dates."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    df = pd.read_csv(csv_path)
    required = REFERENCE_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError(f"{csv_path.name} contains invalid Date values.")
    if df[required].isna().any().any():
        raise ValueError(f"{csv_path.name} contains missing values.")
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


def get_xy(df: pd.DataFrame, target: str):
    """Split DataFrame into X features and y target."""
    return df[FEATURE_COLUMNS].copy(), df[target].copy()


# ── Statistical metrics ────────────────────────────────────────────────────────

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where predicted sign matches true sign."""
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def compute_statistical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and directional accuracy."""
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


# ── Backtest metrics ───────────────────────────────────────────────────────────

def compute_sharpe_ratio(daily_returns: np.ndarray) -> float:
    """
    Annualized Sharpe ratio. Assumes risk-free rate of 0.
    Sharpe = (mean daily return / std of daily returns) * sqrt(252)
    """
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum peak-to-trough decline as a fraction.
    Returns negative number — e.g. -0.15 means 15% drawdown.
    """
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(drawdowns.min())


def run_backtest(
    df: pd.DataFrame, predictions: np.ndarray, target: str,
) -> dict:
    """
    Run two backtest strategies and return performance metrics.

    Strategy 1 — Long Only:
        Each day, equally weight the top K predicted stocks.
        Earn the average actual return of those stocks.

    Strategy 2 — Long-Short:
        Each day, long top K predicted stocks, short bottom K.
        Earn (long return - short return) / 2 (dollar-neutral).

    Both compared to buy-and-hold equal-weighted portfolio of all 30 stocks.
    """
    df = df.copy()
    df["y_pred"] = predictions
    df["y_true"] = df[target]

    # Group by date — for each day we need predictions for all stocks
    daily_returns_long_only  = []
    daily_returns_long_short = []
    daily_returns_buyhold    = []
    dates = []

    for date, group in df.groupby("Date"):
        if len(group) < TOP_K_LONGS + BOTTOM_K_SHORTS:
            continue  # skip days with too few stocks

        # Sort by predicted return (highest first)
        sorted_group = group.sort_values("y_pred", ascending=False)

        # Long top K, short bottom K
        top_k    = sorted_group.head(TOP_K_LONGS)
        bottom_k = sorted_group.tail(BOTTOM_K_SHORTS)

        long_return  = top_k["y_true"].mean()
        short_return = bottom_k["y_true"].mean()

        # Long-only: simply hold the longs
        daily_returns_long_only.append(long_return)

        # Long-short: profit from longs going up AND shorts going down
        # Dollar-neutral: half capital long, half short
        long_short_return = (long_return - short_return) / 2.0
        daily_returns_long_short.append(long_short_return)

        # Buy-and-hold benchmark: equal-weight all stocks
        daily_returns_buyhold.append(group["y_true"].mean())

        dates.append(date)

    if len(daily_returns_long_only) == 0:
        return {"error": "Not enough data for backtest"}

    # Convert log returns to simple returns for compounding
    long_only_simple   = np.exp(np.array(daily_returns_long_only))   - 1
    long_short_simple  = np.exp(np.array(daily_returns_long_short))  - 1
    buyhold_simple     = np.exp(np.array(daily_returns_buyhold))     - 1

    # Build equity curves starting from initial capital
    long_only_equity  = INITIAL_CAPITAL * np.cumprod(1 + long_only_simple)
    long_short_equity = INITIAL_CAPITAL * np.cumprod(1 + long_short_simple)
    buyhold_equity    = INITIAL_CAPITAL * np.cumprod(1 + buyhold_simple)

    return {
        "long_only": {
            "final_value":   float(long_only_equity[-1]),
            "total_return":  float(long_only_equity[-1] / INITIAL_CAPITAL - 1),
            "sharpe_ratio":  compute_sharpe_ratio(long_only_simple),
            "max_drawdown":  compute_max_drawdown(long_only_equity),
            "n_trading_days": len(long_only_simple),
        },
        "long_short": {
            "final_value":   float(long_short_equity[-1]),
            "total_return":  float(long_short_equity[-1] / INITIAL_CAPITAL - 1),
            "sharpe_ratio":  compute_sharpe_ratio(long_short_simple),
            "max_drawdown":  compute_max_drawdown(long_short_equity),
            "n_trading_days": len(long_short_simple),
        },
        "buy_and_hold_benchmark": {
            "final_value":   float(buyhold_equity[-1]),
            "total_return":  float(buyhold_equity[-1] / INITIAL_CAPITAL - 1),
            "sharpe_ratio":  compute_sharpe_ratio(buyhold_simple),
            "max_drawdown":  compute_max_drawdown(buyhold_equity),
        },
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "top_k_longs":     TOP_K_LONGS,
            "bottom_k_shorts": BOTTOM_K_SHORTS,
        },
    }


def compute_per_ticker_metrics(
    df: pd.DataFrame, predictions: np.ndarray, target: str,
) -> dict:
    """Break down statistical metrics by ticker."""
    df = df.copy()
    df["y_pred"] = predictions
    per_ticker = {}
    for ticker, group in df.groupby("ticker"):
        y_true = group[target].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        per_ticker[ticker] = compute_statistical_metrics(y_true, y_pred)
    return per_ticker


# ── Predictors ─────────────────────────────────────────────────────────────────

def historical_mean_predictor(
    train_df: pd.DataFrame, target: str, n: int,
) -> np.ndarray:
    """Predict every value as the mean target from training set only."""
    return np.full(n, train_df[target].mean(), dtype=float)


def last_return_predictor(df: pd.DataFrame) -> np.ndarray:
    """Predict tomorrow's return as today's log_return. 1-day target only."""
    return df["log_return"].to_numpy(dtype=float)


def build_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit scaler on training data only — never on val or test."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ── Saving ─────────────────────────────────────────────────────────────────────

def save_predictions(
    df: pd.DataFrame, predictions: np.ndarray,
    model_name: str, split_name: str, target: str,
) -> None:
    """Save predictions alongside date, ticker, and true target."""
    out = df[REFERENCE_COLUMNS].copy()
    out["y_true"] = df[target].values
    out["y_pred"] = predictions
    suffix = "1d" if "next_day" in target else "5d"
    path = PREDICTIONS_DIR / f"{model_name}_{suffix}_{split_name}_predictions.csv"
    out.to_csv(path, index=False)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str, target: str,
    train_df: pd.DataFrame, train_preds: np.ndarray,
    val_df:   pd.DataFrame, val_preds:   np.ndarray,
    test_df:  pd.DataFrame, test_preds:  np.ndarray,
) -> dict:
    """
    Evaluate on all three splits.
    Backtest is run ONLY on the test set — we don't backtest training data.
    """
    return {
        "model":           model_name,
        "target":          target,
        "train":           compute_statistical_metrics(train_df[target].to_numpy(), train_preds),
        "validation":      compute_statistical_metrics(val_df[target].to_numpy(),   val_preds),
        "test":            compute_statistical_metrics(test_df[target].to_numpy(),  test_preds),
        "test_per_ticker": compute_per_ticker_metrics(test_df, test_preds, target),
        "test_backtest":   run_backtest(test_df, test_preds, target),
    }


# ── Model training ─────────────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Train a simple linear regression baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, n_samples: int,
) -> RandomForestRegressor:
    """
    Train Random Forest with time-weighted samples.
    Recent training data receives more weight (1.0) than older data (0.5),
    because market patterns change over time.
    """
    weights = np.linspace(0.5, 1.0, n_samples)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=weights)
    return model


# ── Printing ───────────────────────────────────────────────────────────────────

def print_summary(results: list, target_label: str) -> None:
    """Print statistical metrics and backtest results for one target."""
    print(f"\n══ {target_label} ══════════════════════════════════════")

    # Statistical metrics
    print(f"\n  Statistical Metrics:")
    for r in results:
        print(f"  {r['model']}")
        for split in ("train", "validation", "test"):
            m = r[split]
            print(
                f"    {split.capitalize():<12} -> "
                f"MAE: {m['mae']:.6f}  "
                f"RMSE: {m['rmse']:.6f}  "
                f"Dir Acc: {m['directional_accuracy']:.4f}"
            )

    # Backtest results
    print(f"\n  Backtest on Test Set (starting ${INITIAL_CAPITAL:,.0f}):")
    print(f"  {'Model':<22} {'Long Only':<22} {'Long-Short':<22}")
    print(f"  {'─'*22} {'─'*22} {'─'*22}")
    for r in results:
        bt = r.get("test_backtest", {})
        if "error" in bt:
            continue
        lo = bt["long_only"]
        ls = bt["long_short"]
        lo_str = f"${lo['final_value']:>9,.0f} ({lo['total_return']*100:+.1f}%)"
        ls_str = f"${ls['final_value']:>9,.0f} ({ls['total_return']*100:+.1f}%)"
        print(f"  {r['model']:<22} {lo_str:<22} {ls_str:<22}")

    # Sharpe and drawdown
    print(f"\n  Risk-Adjusted Metrics:")
    print(f"  {'Model':<22} {'Sharpe (LO)':<14} {'Sharpe (LS)':<14} {'MaxDD (LO)':<14} {'MaxDD (LS)':<14}")
    print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*14} {'─'*14}")
    for r in results:
        bt = r.get("test_backtest", {})
        if "error" in bt:
            continue
        lo = bt["long_only"]
        ls = bt["long_short"]
        print(
            f"  {r['model']:<22} "
            f"{lo['sharpe_ratio']:<14.3f} "
            f"{ls['sharpe_ratio']:<14.3f} "
            f"{lo['max_drawdown']*100:>+10.1f}%   "
            f"{ls['max_drawdown']*100:>+10.1f}%"
        )

    # Benchmark
    if results:
        bt = results[0].get("test_backtest", {})
        if "buy_and_hold_benchmark" in bt:
            bh = bt["buy_and_hold_benchmark"]
            print(f"\n  Buy-and-Hold Benchmark (equal-weight all 30 stocks):")
            print(
                f"    Final value: ${bh['final_value']:,.0f}  "
                f"({bh['total_return']*100:+.1f}%)  "
                f"Sharpe: {bh['sharpe_ratio']:.3f}  "
                f"MaxDD: {bh['max_drawdown']*100:+.1f}%"
            )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Train baselines on both targets and evaluate with backtest."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    all_results = []

    for target in TARGET_COLUMNS:
        target_label = "1-DAY TARGET" if "next_day" in target else "5-DAY TARGET"

        X_train, y_train = get_xy(train_df, target)
        X_val,   y_val   = get_xy(val_df,   target)
        X_test,  y_test  = get_xy(test_df,  target)

        scaler         = build_scaler(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)

        results = []

        # ── 1. Historical Mean ────────────────────────────────────────────────
        hist_train = historical_mean_predictor(train_df, target, len(train_df))
        hist_val   = historical_mean_predictor(train_df, target, len(val_df))
        hist_test  = historical_mean_predictor(train_df, target, len(test_df))

        save_predictions(val_df,  hist_val,  "historical_mean", "val",  target)
        save_predictions(test_df, hist_test, "historical_mean", "test", target)

        results.append(evaluate_model(
            "historical_mean", target,
            train_df, hist_train, val_df, hist_val, test_df, hist_test,
        ))

        # ── 2. Naive Last Return (1-day only) ────────────────────────────────
        if "next_day" in target:
            naive_train = last_return_predictor(train_df)
            naive_val   = last_return_predictor(val_df)
            naive_test  = last_return_predictor(test_df)

            save_predictions(val_df,  naive_val,  "naive_last_return", "val",  target)
            save_predictions(test_df, naive_test, "naive_last_return", "test", target)

            results.append(evaluate_model(
                "naive_last_return", target,
                train_df, naive_train, val_df, naive_val, test_df, naive_test,
            ))

        # ── 3. Linear Regression ─────────────────────────────────────────────
        lr_model = train_linear_regression(X_train_scaled, y_train)
        lr_train = lr_model.predict(X_train_scaled)
        lr_val   = lr_model.predict(X_val_scaled)
        lr_test  = lr_model.predict(X_test_scaled)

        save_predictions(val_df,  lr_val,  "linear_regression", "val",  target)
        save_predictions(test_df, lr_test, "linear_regression", "test", target)

        results.append(evaluate_model(
            "linear_regression", target,
            train_df, lr_train, val_df, lr_val, test_df, lr_test,
        ))

        # ── 4. Random Forest (time-weighted) ─────────────────────────────────
        rf_model = train_random_forest(X_train_scaled, y_train, len(X_train_scaled))
        rf_train = rf_model.predict(X_train_scaled)
        rf_val   = rf_model.predict(X_val_scaled)
        rf_test  = rf_model.predict(X_test_scaled)

        save_predictions(val_df,  rf_val,  "random_forest", "val",  target)
        save_predictions(test_df, rf_test, "random_forest", "test", target)

        results.append(evaluate_model(
            "random_forest", target,
            train_df, rf_train, val_df, rf_val, test_df, rf_test,
        ))

        print_summary(results, target_label)
        all_results.extend(results)

    # ── Save all metrics ──────────────────────────────────────────────────────
    metrics_path = METRICS_DIR / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Saved metrics     : {metrics_path}")
    print(f"  Saved predictions : {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
