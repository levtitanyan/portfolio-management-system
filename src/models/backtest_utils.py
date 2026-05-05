"""
Shared backtest helpers for model evaluation scripts.

The strategy backtests compound simple portfolio returns. For a 5-day target,
the portfolio is rebalanced every fifth prediction date and the non-overlapping
5-day realized return is used for that holding period.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def infer_holding_days(target: str, holding_days: int | None = None) -> int:
    """Infer rebalance spacing from the target name unless explicitly supplied."""
    if holding_days is not None:
        if holding_days < 1:
            raise ValueError("holding_days must be >= 1")
        return int(holding_days)
    target_text = str(target).lower()
    if "30d" in target_text:
        return 30
    if "10d" in target_text:
        return 10
    if "5d" in target_text:
        return 5
    return 1


def max_drawdown(equity: Iterable[float]) -> float:
    """Worst peak-to-trough drawdown for an equity curve."""
    values = np.asarray(list(equity), dtype=float)
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    return float(((values - peak) / peak).min())


def summarize_period_returns(
    period_returns: Iterable[float],
    *,
    initial_capital: float,
    holding_days: int,
    trading_days_per_year: int = 252,
) -> dict:
    """Compound period returns and return the common report metrics."""
    returns = np.asarray(list(period_returns), dtype=float)
    if len(returns) == 0:
        return {}

    equity_after_periods = initial_capital * np.cumprod(1.0 + returns)
    equity_with_initial = np.concatenate([[initial_capital], equity_after_periods])
    periods_per_year = trading_days_per_year / holding_days

    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    return {
        "final_value": float(equity_after_periods[-1]),
        "total_return": float(equity_after_periods[-1] / initial_capital - 1.0),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(equity_with_initial),
        "n_trading_days": int(len(returns) * holding_days),
        "n_periods": int(len(returns)),
        "holding_days": int(holding_days),
    }


def summarize_equity_curve(
    equity: Iterable[float],
    *,
    initial_capital: float,
    trading_days_per_year: int = 252,
) -> dict:
    """Summarize an already-built daily equity curve."""
    values = np.asarray(list(equity), dtype=float)
    if len(values) == 0:
        return {}
    if len(values) == 1:
        return {
            "final_value": float(values[-1]),
            "total_return": float(values[-1] / initial_capital - 1.0),
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "n_trading_days": 0,
            "n_periods": 0,
            "holding_days": 1,
        }

    returns = values[1:] / values[:-1] - 1.0
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(trading_days_per_year)) if std > 0 else 0.0

    return {
        "final_value": float(values[-1]),
        "total_return": float(values[-1] / initial_capital - 1.0),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(values),
        "n_trading_days": int(len(returns)),
        "n_periods": int(len(returns)),
        "holding_days": 1,
    }


def static_buy_and_hold_benchmark(
    df: pd.DataFrame,
    *,
    initial_capital: float,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Equal-dollar buy-and-hold benchmark using adjusted close prices.

    This buys the available stock universe at the first date and holds those
    shares through the last date. It is intentionally independent of whether
    the model target is 1-day or 5-day.
    """
    if "adj_close" not in df.columns:
        return {}

    work = df[["Date", "ticker", "adj_close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    prices = (
        work.pivot_table(index="Date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
        .dropna(axis=1, how="any")
    )

    if prices.shape[0] < 2 or prices.shape[1] == 0:
        return {}

    relative_values = prices.divide(prices.iloc[0])
    equity = initial_capital * relative_values.mean(axis=1)
    return summarize_equity_curve(
        equity.values,
        initial_capital=initial_capital,
        trading_days_per_year=trading_days_per_year,
    )


def daily_equal_weight_benchmark(
    df: pd.DataFrame,
    *,
    initial_capital: float,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Fallback benchmark from next-day returns when prices are unavailable.

    This is a daily equal-weight market benchmark, not a true share-based
    buy-and-hold benchmark. It is used only when adj_close is missing.
    """
    if "target_next_day_return" not in df.columns:
        return {}

    work = df[["Date", "ticker", "target_next_day_return"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    period_returns = []
    for _, group in work.sort_values(["Date", "ticker"]).groupby("Date", sort=True):
        simple_returns = np.expm1(group["target_next_day_return"].astype(float).to_numpy())
        period_returns.append(float(simple_returns.mean()))

    return summarize_period_returns(
        period_returns,
        initial_capital=initial_capital,
        holding_days=1,
        trading_days_per_year=trading_days_per_year,
    )


def run_model_backtest(
    df: pd.DataFrame,
    preds: Iterable[float],
    target: str,
    *,
    holding_days: int | None = None,
    benchmark_df: pd.DataFrame | None = None,
    initial_capital: float,
    top_k_longs: int,
    bottom_k_shorts: int,
    trading_days_per_year: int = 252,
    cost_per_trade: float = 0.001,
) -> dict:
    """Run long-only, long-short, and horizon-independent benchmark backtests.

    cost_per_trade is applied each rebalancing period as a fraction of portfolio
    value (0.001 = 0.1%). This approximates round-trip transaction costs under
    the assumption of full portfolio turnover each period — a conservative
    estimate consistent with the 0.1% per-trade commitment in the paper.
    """
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["y_pred"] = np.asarray(list(preds), dtype=float)
    work["y_true"] = work[target].astype(float)

    hold = infer_holding_days(target, holding_days)
    dates = sorted(work["Date"].unique())
    rebal_dates = dates[::hold]

    long_only_returns: list[float] = []
    long_short_returns: list[float] = []

    min_stocks = top_k_longs + bottom_k_shorts
    for rebal_date in rebal_dates:
        day_df = work[work["Date"] == rebal_date].copy()
        if len(day_df) < min_stocks:
            continue

        ranked = day_df.sort_values("y_pred", ascending=False)
        long_simple = np.expm1(ranked.head(top_k_longs)["y_true"].to_numpy(dtype=float))
        short_simple = np.expm1(ranked.tail(bottom_k_shorts)["y_true"].to_numpy(dtype=float))

        long_return = float(long_simple.mean())
        short_return = float(short_simple.mean())
        long_only_returns.append(long_return - cost_per_trade)
        long_short_returns.append((long_return - short_return) / 2.0 - cost_per_trade)

    if not long_only_returns:
        return {"error": "Not enough data"}

    # Align benchmark to the strategy's actual trading window so returns are
    # comparable. Without this, the benchmark starts at the first test date
    # while the strategy starts at its first rebalancing date (which may be
    # the same day but the strategy's last period can extend further).
    benchmark_source = benchmark_df.copy() if benchmark_df is not None else work
    benchmark_source["Date"] = pd.to_datetime(benchmark_source["Date"])
    benchmark_source = benchmark_source[benchmark_source["Date"] >= rebal_dates[0]]
    buy_hold = static_buy_and_hold_benchmark(
        benchmark_source,
        initial_capital=initial_capital,
        trading_days_per_year=trading_days_per_year,
    )
    if not buy_hold:
        buy_hold = daily_equal_weight_benchmark(
            benchmark_source,
            initial_capital=initial_capital,
            trading_days_per_year=trading_days_per_year,
        )

    return {
        "long_only": summarize_period_returns(
            long_only_returns,
            initial_capital=initial_capital,
            holding_days=hold,
            trading_days_per_year=trading_days_per_year,
        ),
        "long_short": summarize_period_returns(
            long_short_returns,
            initial_capital=initial_capital,
            holding_days=hold,
            trading_days_per_year=trading_days_per_year,
        ),
        "buy_and_hold_benchmark": buy_hold,
    }
