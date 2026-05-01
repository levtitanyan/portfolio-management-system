"""
Portfolio Backtest
==================
Converts model predictions into a trading strategy and evaluates
portfolio performance with realistic transaction costs.

Strategies:
    Long-Only:   Buy equal-weight top K predicted stocks each period
    Long-Short:  Long top K, short bottom K (dollar-neutral)
    Buy-and-Hold: Equal-weight all stocks, never rebalance

Models evaluated:
    5-day: random_forest, ridge_regression, gru_default (best performers)
    1-day: random_forest, ridge_regression, lstm_default

Output:
    outputs/portfolio/
        equity_curves_5d.csv      — daily portfolio values for all strategies
        equity_curves_1d.csv
        metrics_5d.csv            — full performance metrics per model
        metrics_1d.csv
        trades_5d.csv             — every trade with cost breakdown
        trades_1d.csv
        summary_report.md         — readable summary for paper

Run:
    python src/models/portfolio_backtest.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

def find_base_dir() -> Path:
    """Find the project root whether this file is run from root or src/models."""
    file_path = Path(__file__).resolve()
    for parent in [file_path.parent, *file_path.parents]:
        if (parent / "outputs").exists() and (parent / "data").exists():
            return parent
    return file_path.parent


BASE_DIR     = find_base_dir()
OUTPUTS_DIR  = BASE_DIR / "outputs"
PORTFOLIO_DIR = OUTPUTS_DIR / "portfolio"

# ── Configuration ──────────────────────────────────────────────────────────────

INITIAL_CAPITAL       = 10_000.0
TOP_K_LONGS           = 5
BOTTOM_K_SHORTS       = 5
COST_PER_TRADE        = 0.001   # 10 bps per $1 of portfolio turnover
MAX_POSITION_WEIGHT   = 0.30    # no single stock > 30% of portfolio
DRAWDOWN_THRESHOLD    = 0.15    # reduce sizes 50% if drawdown exceeds 15%
TRADING_DAYS_PER_YEAR = 252

# Models to backtest per horizon
MODELS_5D = ["random_forest", "ridge_regression", "gru_default"]
MODELS_1D = ["random_forest", "ridge_regression", "lstm_default"]

# ── Prediction file discovery ──────────────────────────────────────────────────

def find_predictions(model_name: str, horizon: str) -> Path | None:
    """
    Find the test prediction CSV for a given model and horizon.
    Searches outputs/ recursively for predictions_{horizon}_test.csv
    inside a folder matching the model name.
    """
    pattern = f"predictions_{horizon}_test.csv"

    for path in OUTPUTS_DIR.rglob(pattern):
        if "reports" in str(path):
            continue
        parts = list(path.relative_to(OUTPUTS_DIR).parts)

        # baselines/random_forest/predictions_5d_test.csv
        if len(parts) >= 2 and parts[0] == "baselines" and parts[1] == model_name:
            return path

        # lstm/default/predictions_5d_test.csv  → model name = lstm_default
        if len(parts) >= 3:
            folder_name = f"{parts[0]}_{parts[1]}"
            if folder_name == model_name:
                return path

        # direct match on parent folder name
        if path.parent.name == model_name:
            return path

    return None


def load_predictions(model_name: str, horizon: str) -> pd.DataFrame | None:
    """Load and validate a prediction CSV."""
    path = find_predictions(model_name, horizon)
    if path is None:
        print(f"  [SKIP] {model_name} ({horizon}) — prediction file not found")
        return None

    df = pd.read_csv(path)
    required = {"Date", "ticker", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        print(f"  [SKIP] {model_name} ({horizon}) — missing columns in {path}")
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)
    print(f"  [OK]   {model_name} ({horizon}) — {len(df)} rows, "
          f"{df['Date'].nunique()} dates, {df['ticker'].nunique()} tickers")
    return df


# ── Portfolio construction ─────────────────────────────────────────────────────

def build_portfolio(
    df: pd.DataFrame,
    strategy: str,           # "long_only" or "long_short"
    holding_days: int,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a portfolio from predictions.

    Rebalances every `holding_days` trading dates.
    Applies transaction costs, position limits, and drawdown control.

    Returns:
        equity_df : one row per rebalancing date with portfolio value
        trades_df : one row per trade (buy or sell)
    """
    dates      = sorted(df["Date"].unique())
    capital    = initial_capital
    peak       = initial_capital
    equity_rows = []
    trade_rows  = []
    current_weights: dict[str, float] = {}  # ticker -> portfolio weight

    # Rebalance every holding_days dates
    rebal_dates = dates[::holding_days]

    for i, rebal_date in enumerate(rebal_dates):
        day_df = df[df["Date"] == rebal_date].copy()

        if len(day_df) < TOP_K_LONGS + (BOTTOM_K_SHORTS if strategy == "long_short" else 0):
            # Not enough stocks — carry forward
            equity_rows.append({
                "Date":      rebal_date,
                "portfolio_value": capital,
                "strategy":  strategy,
            })
            continue

        day_df = day_df.sort_values("y_pred", ascending=False).reset_index(drop=True)
        longs  = day_df.head(TOP_K_LONGS)["ticker"].tolist()

        if strategy == "long_short":
            shorts = day_df.tail(BOTTOM_K_SHORTS)["ticker"].tolist()
        else:
            shorts = []

        # ── Drawdown control ───────────────────────────────────────────────────
        drawdown  = (capital - peak) / peak if peak > 0 else 0
        size_mult = 0.5 if drawdown < -DRAWDOWN_THRESHOLD else 1.0

        # ── Target weights and transaction costs ───────────────────────────────
        # Transaction cost is charged on traded notional, not on position count.
        if strategy == "long_only":
            target_weights = {ticker: size_mult / TOP_K_LONGS for ticker in longs}
        else:
            target_weights = {ticker: 0.5 * size_mult / TOP_K_LONGS for ticker in longs}
            target_weights.update({
                ticker: -0.5 * size_mult / BOTTOM_K_SHORTS for ticker in shorts
            })

        turnover = sum(
            abs(target_weights.get(ticker, 0.0) - current_weights.get(ticker, 0.0))
            for ticker in set(target_weights) | set(current_weights)
        )
        cost_rate = COST_PER_TRADE * turnover

        # ── Apply realized return ──────────────────────────────────────────────
        simple_returns = {
            row.ticker: float(np.exp(row.y_true) - 1)
            for row in day_df.itertuples(index=False)
        }
        gross_return = float(
            sum(weight * simple_returns.get(ticker, 0.0)
                for ticker, weight in target_weights.items())
        )

        if strategy == "long_only":
            long_returns = float(np.mean([simple_returns[ticker] for ticker in longs]))
            short_returns = None
        else:
            long_returns = float(np.mean([simple_returns[ticker] for ticker in longs]))
            short_returns = float(np.mean([simple_returns[ticker] for ticker in shorts]))

        net_return = gross_return - cost_rate
        capital    = capital * (1 + net_return)
        peak       = max(peak, capital)

        current_weights = target_weights

        equity_rows.append({
            "Date":            rebal_date,
            "portfolio_value": capital,
            "strategy":        strategy,
            "long_return":     long_returns,
            "short_return":    short_returns if shorts else None,
            "gross_return":    gross_return,
            "net_return":      net_return,
            "cost":            cost_rate,
            "turnover":        turnover,
            "drawdown":        drawdown,
            "size_mult":       size_mult,
        })

        trade_rows.append({
            "Date":          rebal_date,
            "longs":         ",".join(longs),
            "shorts":        ",".join(shorts),
            "turnover":      turnover,
            "cost_rate":     cost_rate,
            "gross_return":  gross_return,
            "net_return":    net_return,
            "capital_after": capital,
        })

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    return equity_df, trades_df


def build_buy_hold(df: pd.DataFrame, holding_days: int, initial_capital: float) -> pd.DataFrame:
    """
    Equal-weight buy-and-hold baseline.
    Buys all stocks on day 1, holds until end.
    Portfolio value tracks average cumulative return.
    """
    dates = sorted(df["Date"].unique())
    rebal_dates = dates[::holding_days]

    # Compute cumulative log returns per date
    rows = []
    capital = initial_capital

    for rebal_date in rebal_dates:
        day_df = df[df["Date"] == rebal_date]
        avg_return = float(day_df["y_true"].mean()) if len(day_df) > 0 else 0.0
        period_return = float(np.exp(avg_return) - 1)
        capital = capital * (1 + period_return)
        rows.append({
            "Date":            rebal_date,
            "portfolio_value": capital,
            "strategy":        "buy_and_hold",
            "gross_return":    period_return,
            "net_return":      period_return,
            "cost":            0.0,
            "turnover":        0.0,
        })

    return pd.DataFrame(rows)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_portfolio_metrics(
    equity_df: pd.DataFrame,
    initial_capital: float,
    holding_days: int = 1,
) -> dict:
    """Compute comprehensive portfolio metrics from equity curve."""
    if equity_df.empty or "portfolio_value" not in equity_df.columns:
        return {}

    values = equity_df["portfolio_value"].values
    if "net_return" in equity_df.columns:
        returns = equity_df["net_return"].dropna().to_numpy(dtype=float)
    else:
        returns = np.diff(values) / values[:-1]

    final_value    = float(values[-1])
    total_return   = float(final_value / initial_capital - 1)
    n_periods      = len(returns)
    periods_per_yr = TRADING_DAYS_PER_YEAR / max(1, holding_days)

    # Annualized return
    if n_periods > 0 and values[0] > 0:
        years = n_periods / periods_per_yr
        ann_return = float((final_value / initial_capital) ** (1 / years) - 1)
    else:
        ann_return = total_return

    # Sharpe ratio
    return_std = returns.std(ddof=1) if len(returns) > 1 else 0
    if len(returns) > 1 and return_std > 0:
        sharpe = float(returns.mean() / return_std * np.sqrt(periods_per_yr))
    else:
        sharpe = 0.0

    # Max drawdown
    curve_values = np.concatenate(([initial_capital], values))
    peak    = np.maximum.accumulate(curve_values)
    dd      = (curve_values - peak) / peak
    max_dd  = float(dd.min())

    # Calmar ratio
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else float("nan")

    # Win rate (% of periods with positive return)
    win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

    # Significance test
    if len(returns) > 10:
        _, p_value = ttest_1samp(returns, 0)
    else:
        p_value = float("nan")

    total_costs   = float(equity_df["cost"].sum()) if "cost" in equity_df.columns else 0.0

    return {
        "final_value":      final_value,
        "total_return":     total_return,
        "annualized_return": ann_return,
        "sharpe_ratio":     sharpe,
        "max_drawdown":     max_dd,
        "calmar_ratio":     calmar,
        "win_rate":         win_rate,
        "n_periods":        n_periods,
        "total_costs":      total_costs,
        "p_value":          p_value,
        "significant":      p_value < 0.05 if not pd.isna(p_value) else False,
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_metrics(model_name: str, strategy: str, horizon: str, m: dict, bh: dict) -> None:
    ret    = m.get("total_return", 0)
    sharpe = m.get("sharpe_ratio", 0)
    dd     = m.get("max_drawdown", 0)
    calmar = m.get("calmar_ratio", float("nan"))
    win    = m.get("win_rate", 0)
    cost   = m.get("total_costs", 0)
    sig    = "✓" if m.get("significant") else "✗"
    excess = ret - bh.get("total_return", 0)

    print(f"    Return:     {ret:+.1%}  (excess vs B&H: {excess:+.1%})")
    print(f"    Sharpe:     {sharpe:.3f}")
    print(f"    Max DD:     {dd:.1%}")
    print(f"    Calmar:     {calmar:.2f}" if not pd.isna(calmar) else "    Calmar:     —")
    print(f"    Win rate:   {win:.1%}")
    print(f"    Total cost: {cost:.4f}")
    print(f"    Significant:{sig} (p={m.get('p_value', float('nan')):.4f})")


def build_summary_report(all_results: list[dict]) -> str:
    """Build a Markdown summary report for the paper."""
    lines = [
        "# Portfolio Backtest Results",
        "",
        f"Initial capital: ${INITIAL_CAPITAL:,.0f}",
        f"Transaction cost: {COST_PER_TRADE:.1%} per $1 of portfolio turnover",
        f"Long positions: top {TOP_K_LONGS} stocks",
        f"Short positions: bottom {BOTTOM_K_SHORTS} stocks (long-short only)",
        f"Drawdown control: reduce size by 50% if drawdown exceeds {DRAWDOWN_THRESHOLD:.0%}",
        "",
    ]

    for horizon in ["5d", "1d"]:
        label = "5-Day" if horizon == "5d" else "1-Day"
        lines.append(f"## {label} Horizon")
        lines.append("")
        lines.append("| Model | Strategy | Return | Sharpe | Max DD | Calmar | vs B&H | Win% | Sig |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        horizon_results = [r for r in all_results if r["horizon"] == horizon]
        bh_return = next(
            (r["metrics"]["total_return"] for r in horizon_results
             if r["model"] == "buy_and_hold"), 0.0
        )

        for r in sorted(horizon_results, key=lambda x: x["metrics"].get("total_return", 0), reverse=True):
            m   = r["metrics"]
            ret = m.get("total_return", 0)
            sh  = m.get("sharpe_ratio", 0)
            dd  = m.get("max_drawdown", 0)
            cal = m.get("calmar_ratio", float("nan"))
            win = m.get("win_rate", 0)
            exc = ret - bh_return if r["model"] != "buy_and_hold" else 0
            sig = "✓" if m.get("significant") else "—"
            cal_s = f"{cal:.2f}" if not pd.isna(cal) else "—"

            lines.append(
                f"| {r['model']} | {r['strategy']} | {ret:+.1%} | {sh:.3f} | "
                f"{dd:.1%} | {cal_s} | {exc:+.1%} | {win:.1%} | {sig} |"
            )

        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- Returns compound over each holding period",
        "- Transaction costs are charged on portfolio turnover, not position count",
        "- Drawdown control activates at -15% portfolio drawdown",
        "- Significance test: one-sample t-test on period returns (H0: mean=0)",
        "- Buy-and-hold: equal-weight all stocks, rebalanced each period",
        "- 2024 was a strong bull market — results may not generalize to bear markets",
        "",
    ])

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    all_results       = []
    all_equity_rows   = []
    all_trades_rows   = []
    all_metrics_rows  = []

    config = {
        "1d": {"models": MODELS_1D, "holding_days": 1},
        "5d": {"models": MODELS_5D, "holding_days": 5},
    }

    for horizon, cfg in config.items():
        holding = cfg["holding_days"]
        label   = "5-Day" if horizon == "5d" else "1-Day"
        print(f"\n{'='*60}")
        print(f"  {label} Horizon  (holding period = {holding} day(s))")
        print(f"{'='*60}")

        # ── Buy-and-hold baseline (use first available model's predictions) ───
        bh_metrics = {}
        for model_name in cfg["models"]:
            df = load_predictions(model_name, horizon)
            if df is not None:
                bh_equity  = build_buy_hold(df, holding, INITIAL_CAPITAL)
                bh_metrics = compute_portfolio_metrics(
                    bh_equity, INITIAL_CAPITAL, holding
                )

                bh_equity["model"]   = "buy_and_hold"
                bh_equity["horizon"] = horizon
                all_equity_rows.append(bh_equity)

                all_results.append({
                    "model":    "buy_and_hold",
                    "strategy": "buy_and_hold",
                    "horizon":  horizon,
                    "metrics":  bh_metrics,
                })
                break

        print(f"\n  Buy-and-Hold: {bh_metrics.get('total_return', 0):+.1%}")

        # ── Each model ─────────────────────────────────────────────────────────
        for model_name in cfg["models"]:
            print(f"\n  Model: {model_name}")
            df = load_predictions(model_name, horizon)
            if df is None:
                continue

            for strategy in ["long_only", "long_short"]:
                print(f"\n    Strategy: {strategy}")
                equity_df, trades_df = build_portfolio(
                    df, strategy, holding, INITIAL_CAPITAL
                )

                if equity_df.empty:
                    print("    [SKIP] No equity data produced")
                    continue

                metrics = compute_portfolio_metrics(
                    equity_df, INITIAL_CAPITAL, holding
                )
                print_metrics(model_name, strategy, horizon, metrics, bh_metrics)

                # Tag for saving
                equity_df["model"]    = model_name
                equity_df["strategy"] = strategy
                equity_df["horizon"]  = horizon

                trades_df["model"]    = model_name
                trades_df["strategy"] = strategy
                trades_df["horizon"]  = horizon

                all_equity_rows.append(equity_df)
                all_trades_rows.append(trades_df)

                all_results.append({
                    "model":    model_name,
                    "strategy": strategy,
                    "horizon":  horizon,
                    "metrics":  metrics,
                })

                all_metrics_rows.append({
                    "horizon":          horizon,
                    "model":            model_name,
                    "strategy":         strategy,
                    "final_value":      metrics.get("final_value"),
                    "total_return":     metrics.get("total_return"),
                    "annualized_return": metrics.get("annualized_return"),
                    "sharpe_ratio":     metrics.get("sharpe_ratio"),
                    "max_drawdown":     metrics.get("max_drawdown"),
                    "calmar_ratio":     metrics.get("calmar_ratio"),
                    "win_rate":         metrics.get("win_rate"),
                    "total_costs":      metrics.get("total_costs"),
                    "n_periods":        metrics.get("n_periods"),
                    "p_value":          metrics.get("p_value"),
                    "significant":      metrics.get("significant"),
                    "excess_vs_bh":     metrics.get("total_return", 0) - bh_metrics.get("total_return", 0),
                })

    # ── Save outputs ───────────────────────────────────────────────────────────

    if all_equity_rows:
        equity_all = pd.concat(all_equity_rows, ignore_index=True)
        for h in ["1d", "5d"]:
            sub = equity_all[equity_all["horizon"] == h]
            if not sub.empty:
                sub.to_csv(PORTFOLIO_DIR / f"equity_curves_{h}.csv", index=False)

    if all_trades_rows:
        trades_all = pd.concat(all_trades_rows, ignore_index=True)
        for h in ["1d", "5d"]:
            sub = trades_all[trades_all["horizon"] == h]
            if not sub.empty:
                sub.to_csv(PORTFOLIO_DIR / f"trades_{h}.csv", index=False)

    if all_metrics_rows:
        metrics_df = pd.DataFrame(all_metrics_rows)
        for h in ["1d", "5d"]:
            sub = metrics_df[metrics_df["horizon"] == h].sort_values(
                "total_return", ascending=False
            )
            if not sub.empty:
                sub.to_csv(PORTFOLIO_DIR / f"metrics_{h}.csv", index=False)

    # ── Markdown summary ───────────────────────────────────────────────────────

    report = build_summary_report(all_results)
    report_path = PORTFOLIO_DIR / "summary_report.md"
    report_path.write_text(report, encoding="utf-8")

    # ── Final print ────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("  PORTFOLIO BACKTEST COMPLETE")
    print(f"{'='*60}")

    if all_metrics_rows:
        mdf = pd.DataFrame(all_metrics_rows)
        for h in ["1d", "5d"]:
            sub = mdf[mdf["horizon"] == h].sort_values("total_return", ascending=False)
            if sub.empty: continue
            lbl = "5-Day" if h == "5d" else "1-Day"
            print(f"\n  {lbl} — Top Results (after transaction costs):")
            print(f"  {'Model':<22} {'Strategy':<12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'vs B&H':>10}")
            print(f"  {'─'*22} {'─'*12} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")
            for _, r in sub.head(8).iterrows():
                print(
                    f"  {r['model']:<22} {r['strategy']:<12} "
                    f"{r['total_return']:>+9.1%} {r['sharpe_ratio']:>8.3f} "
                    f"{r['max_drawdown']:>7.1%} {r['excess_vs_bh']:>+9.1%}"
                )

    print(f"\n  Saved to: {PORTFOLIO_DIR}")
    print(f"  Files:")
    for f in sorted(PORTFOLIO_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
