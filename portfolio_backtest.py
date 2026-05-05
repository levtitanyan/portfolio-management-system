"""
Portfolio Backtest
==================
Simulates trading strategies using model predictions and evaluates
portfolio performance with realistic transaction costs.

Strategies per model:
    Long-Only:    Buy equal-weight top K predicted stocks each period
    Long-Short:   Long top K, short bottom K (dollar-neutral)

Benchmarks:
    Buy-and-Hold: Equal-dollar all available stocks, bought once and held

Features:
    - Auto-discovers ALL trained models from outputs/
    - Transaction costs (0.1% per dollar traded)
    - Drawdown control (halves position sizes at -15% drawdown)
    - Full metrics: return, Sharpe, Calmar, max drawdown, win rate
    - Statistical significance test on period returns
    - Equity curves saved as CSV for charting
    - Trade log saved per model

Output structure:
    outputs/{universe}/reports/
        equity_curves_1d.csv       — portfolio value per period per model
        equity_curves_5d.csv
        equity_curves_10d.csv
        equity_curves_30d.csv
        metrics_1d.csv             — all performance metrics, sorted by return
        metrics_5d.csv
        metrics_10d.csv
        metrics_30d.csv
        trades_1d.csv              — every rebalancing trade
        trades_5d.csv
        trades_10d.csv
        trades_30d.csv
        md/backtest_report.md      — markdown summary (sorted by return)

Run:
    python run_all.py --universe tech30
    python run_all.py --universe energy30
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings
import math

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universes import get_data_dir, get_outputs_dir

warnings.filterwarnings("ignore")


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent
OUTPUTS_DIR   = get_outputs_dir()
REPORT_DIR    = OUTPUTS_DIR / "reports"
MD_DIR        = REPORT_DIR / "md"
TEST_PATH     = get_data_dir() / "splits" / "test.csv"


# ── Configuration ──────────────────────────────────────────────────────────────

INITIAL_CAPITAL       = 10_000.0
TOP_K_GRID            = [3, 5, 10, 15]  # tested for each horizon
COST_PER_TRADE        = 0.001    # 0.1% per dollar traded
DRAWDOWN_THRESHOLD    = 0.15     # reduce sizes by 50% if drawdown > 15%
TRADING_DAYS_PER_YEAR = 252
HORIZONS = {
    "1d": {"holding_days": 1, "label": "1-Day"},
    "5d": {"holding_days": 5, "label": "5-Day"},
    "10d": {"holding_days": 10, "label": "10-Day"},
    "30d": {"holding_days": 30, "label": "30-Day"},
}


# ── Model discovery ────────────────────────────────────────────────────────────

def discover_models(horizon: str) -> dict[str, Path]:
    """
    Auto-discover all prediction files for a given horizon.

    Searches the universe outputs directory recursively for
    predictions_{horizon}_test.csv and infers a clean model name.

    Returns dict of {model_name: path}.
    """
    pattern = f"predictions_{horizon}_test.csv"
    found   = {}

    for path in sorted(OUTPUTS_DIR.rglob(pattern)):
        if "reports" in str(path):
            continue

        # Build model name from relative path
        try:
            parts = list(path.relative_to(OUTPUTS_DIR).parts)
        except ValueError:
            continue

        # outputs/baselines/random_forest/predictions_1d_test.csv → random_forest
        if len(parts) >= 3 and parts[0] == "baselines":
            model_name = parts[1]

        # outputs/lstm/default/predictions_1d_test.csv → lstm_default
        # outputs/gru/tuned/predictions_1d_test.csv    → gru_tuned
        elif len(parts) >= 3:
            model_name = f"{parts[0]}_{parts[1]}"

        else:
            model_name = path.parent.name

        found[model_name] = path

    return found


def load_predictions(model_name: str, path: Path) -> pd.DataFrame | None:
    """
    Load and validate a prediction CSV.

    Expected columns: Date, ticker, y_true, y_pred
    y_true = actual return for that period (next-day or 5-day log return)
    y_pred = model's predicted return
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [SKIP] {model_name} — could not read {path}: {e}")
        return None

    required = {"Date", "ticker", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        print(f"  [SKIP] {model_name} — missing columns: {required - set(df.columns)}")
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

    n_dates   = df["Date"].nunique()
    n_tickers = df["ticker"].nunique()
    print(f"  [OK]   {model_name:<25} {len(df):>6} rows  {n_dates:>4} dates  {n_tickers:>3} tickers")

    return df


def load_reference_data() -> pd.DataFrame:
    """Load the full test split with adjusted prices for the benchmark."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark data: {TEST_PATH}")
    df = pd.read_csv(TEST_PATH)
    required = {"Date", "ticker", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark data is missing columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "ticker"]).reset_index(drop=True)


# ── Portfolio simulation ───────────────────────────────────────────────────────

def simulate_portfolio(
    df:              pd.DataFrame,
    strategy:        str,           # "long_only" | "long_short"
    holding_days:    int,
    top_k:           int,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a trading strategy over the test period.

    On each rebalancing date:
      1. Rank stocks by predicted return (y_pred)
      2. Long the top K (long_only) or long top K / short bottom K (long_short)
      3. Earn the actual return (y_true) for those stocks that period
      4. Deduct transaction costs
      5. Apply drawdown control if portfolio is down > 15% from peak

    The y_true column already represents the correct forward return
    (next-day for 1d, 5-day cumulative for 5d) so we use it directly
    as the realized period return without any lookahead.

    Returns:
        equity_df : one row per rebalancing date
        trades_df : one row per rebalancing date with trade details
    """
    dates       = sorted(df["Date"].unique())
    rebal_dates = dates[::holding_days]

    capital      = initial_capital
    peak         = initial_capital
    prev_weights : dict[str, float] = {}

    equity_rows = []
    trade_rows  = []

    for rebal_date in rebal_dates:
        day_df = df[df["Date"] == rebal_date].copy()

        min_stocks = top_k * (2 if strategy == "long_short" else 1)
        if len(day_df) < min_stocks:
            # Not enough stocks available — hold cash
            equity_rows.append({
                "Date":            rebal_date,
                "portfolio_value": capital,
                "gross_return":    0.0,
                "net_return":      0.0,
                "cost":            0.0,
                "drawdown":        (capital - peak) / peak,
                "size_mult":       1.0,
                "n_longs":         0,
                "n_shorts":        0,
                "top_k":           top_k,
            })
            continue

        # ── Stock selection ────────────────────────────────────────────────────
        ranked  = day_df.sort_values("y_pred", ascending=False).reset_index(drop=True)
        longs   = set(ranked.head(top_k)["ticker"].tolist())
        shorts  = set(ranked.tail(top_k)["ticker"].tolist()) if strategy == "long_short" else set()

        # ── Drawdown control ───────────────────────────────────────────────────
        drawdown  = (capital - peak) / peak
        size_mult = 0.5 if drawdown < -DRAWDOWN_THRESHOLD else 1.0

        # ── Target weights ─────────────────────────────────────────────────────
        target_weights: dict[str, float] = {}
        if strategy == "long_only":
            long_weight = size_mult / len(longs)
            target_weights = {ticker: long_weight for ticker in longs}
        else:
            long_weight = size_mult * 0.5 / len(longs)
            short_weight = -size_mult * 0.5 / len(shorts)
            target_weights.update({ticker: long_weight for ticker in longs})
            target_weights.update({ticker: short_weight for ticker in shorts})

        # ── Realized returns ───────────────────────────────────────────────────
        # y_true is the actual log return for that stock and holding period.
        simple_returns = np.expm1(day_df.set_index("ticker")["y_true"].astype(float))
        gross_return = float(
            sum(weight * simple_returns.get(ticker, 0.0)
                for ticker, weight in target_weights.items())
        )

        # ── Transaction costs ──────────────────────────────────────────────────
        # Cost is proportional to turnover, so opening five 20% positions costs
        # 0.1% of capital, not 0.5%.
        all_tickers = set(prev_weights) | set(target_weights)
        turnover = sum(abs(target_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
                       for t in all_tickers)
        cost = COST_PER_TRADE * turnover

        prev_longs = {t for t, w in prev_weights.items() if w > 0}
        prev_shorts = {t for t, w in prev_weights.items() if w < 0}
        new_longs = longs - prev_longs
        new_shorts = shorts - prev_shorts
        closed = set(prev_weights) - set(target_weights)
        n_trades = len(new_longs) + len(new_shorts) + len(closed)

        # ── Net return and capital update ──────────────────────────────────────
        net_return = gross_return - cost
        capital    = capital * (1.0 + net_return)
        peak       = max(peak, capital)

        # Update position tracking
        prev_weights = target_weights

        equity_rows.append({
            "Date":            rebal_date,
            "portfolio_value": capital,
            "gross_return":    gross_return,
            "net_return":      net_return,
            "cost":            cost,
            "drawdown":        (capital - peak) / peak,
            "size_mult":       size_mult,
            "n_longs":         len(longs),
            "n_shorts":        len(shorts),
            "top_k":           top_k,
        })

        trade_rows.append({
            "Date":          rebal_date,
            "longs":         ",".join(sorted(longs)),
            "shorts":        ",".join(sorted(shorts)),
            "n_new_longs":   len(new_longs),
            "n_new_shorts":  len(new_shorts),
            "n_closed":      len(closed),
            "n_trades":      n_trades,
            "turnover":      turnover,
            "cost":          cost,
            "gross_return":  gross_return,
            "net_return":    net_return,
            "capital_after": capital,
            "top_k":         top_k,
        })

    return pd.DataFrame(equity_rows), pd.DataFrame(trade_rows)


def simulate_buy_and_hold(
    df:              pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """
    Equal-dollar buy-and-hold benchmark.
    Buys all stocks once at the first test date and holds those shares.
    This benchmark is the same for 1-day and 5-day model horizons.
    """
    work = df[["Date", "ticker", "adj_close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    prices = (
        work.pivot_table(index="Date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
        .dropna(axis=1, how="any")
    )

    if prices.shape[0] < 2 or prices.shape[1] == 0:
        return pd.DataFrame()

    equity = initial_capital * prices.divide(prices.iloc[0]).mean(axis=1)
    period_returns = equity.pct_change().dropna()
    peak = initial_capital
    equity_rows = []

    for date, period_ret in period_returns.items():
        value = float(equity.loc[date])
        peak = max(peak, value)

        equity_rows.append({
            "Date":            date,
            "portfolio_value": value,
            "gross_return":    float(period_ret),
            "net_return":      float(period_ret),
            "cost":            0.0,
            "drawdown":        (value - peak) / peak,
        })

    return pd.DataFrame(equity_rows)


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_metrics(equity_df: pd.DataFrame, initial_capital: float, holding_days: int = 1) -> dict:
    """
    Compute comprehensive portfolio performance metrics.

    Metrics:
        final_value        — ending portfolio value in dollars
        total_return       — (final - initial) / initial
        annualized_return  — geometric annualized return
        sharpe_ratio       — annualized Sharpe (risk-free rate = 0)
        sortino_ratio      — Sharpe using only downside volatility
        max_drawdown       — worst peak-to-trough decline
        calmar_ratio       — annualized return / |max drawdown|
        win_rate           — fraction of periods with positive net return
        avg_win            — average positive period return
        avg_loss           — average negative period return
        profit_factor      — total gains / total losses
        total_costs        — cumulative transaction costs paid
        n_periods          — number of rebalancing periods
        p_value            — t-test p-value (H0: mean period return = 0)
        significant        — True if p < 0.05
    """
    if equity_df.empty:
        return {}

    values = equity_df["portfolio_value"].values.astype(float)

    if len(values) < 2:
        return {"final_value": float(values[-1]), "total_return": float(values[-1] / initial_capital - 1)}

    # Period returns (net)
    period_rets = equity_df["net_return"].values.astype(float) if "net_return" in equity_df.columns \
                  else np.diff(values) / values[:-1]

    final_value  = float(values[-1])
    total_return = float(final_value / initial_capital - 1)
    n_periods    = len(period_rets)
    periods_per_year = TRADING_DAYS_PER_YEAR / holding_days

    # Annualized return — geometric
    years = n_periods * holding_days / TRADING_DAYS_PER_YEAR
    if years > 0 and initial_capital > 0:
        ann_return = float((final_value / initial_capital) ** (1.0 / years) - 1)
    else:
        ann_return = total_return

    # Sharpe ratio — annualized
    period_std = period_rets.std(ddof=1) if n_periods > 1 else 0.0
    if period_std > 0:
        sharpe = float(period_rets.mean() / period_std * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    # Sortino ratio — uses only downside deviation
    downside = period_rets[period_rets < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_std > 0:
        sortino = float(period_rets.mean() / downside_std * math.sqrt(periods_per_year))
    else:
        sortino = float("nan")

    # Max drawdown
    values_with_initial = np.concatenate([[initial_capital], values])
    peak    = np.maximum.accumulate(values_with_initial)
    dd      = (values_with_initial - peak) / peak
    max_dd  = float(dd.min())

    # Calmar ratio
    calmar = float(ann_return / abs(max_dd)) if max_dd < -1e-6 else float("nan")

    # Win/loss statistics
    wins   = period_rets[period_rets > 0]
    losses = period_rets[period_rets < 0]
    win_rate     = float((period_rets > 0).mean())
    avg_win      = float(wins.mean())   if len(wins) > 0   else 0.0
    avg_loss     = float(losses.mean()) if len(losses) > 0 else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("nan")

    # Total costs
    total_costs = float(equity_df["cost"].sum()) if "cost" in equity_df.columns else 0.0

    # Statistical significance
    if n_periods >= 10:
        _, p_value = ttest_1samp(period_rets, 0)
        significant = bool(p_value < 0.05)
    else:
        p_value     = float("nan")
        significant = False

    return {
        "final_value":        final_value,
        "total_return":       total_return,
        "annualized_return":  ann_return,
        "sharpe_ratio":       sharpe,
        "sortino_ratio":      sortino,
        "max_drawdown":       max_dd,
        "calmar_ratio":       calmar,
        "win_rate":           win_rate,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "profit_factor":      profit_factor,
        "total_costs":        total_costs,
        "n_periods":          n_periods,
        "holding_days":       int(holding_days),
        "n_trading_days":     int(n_periods * holding_days),
        "p_value":            p_value,
        "significant":        significant,
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_metrics(model_name: str, strategy: str, m: dict, bh: dict) -> None:
    ret   = m.get("total_return", 0)
    ann   = m.get("annualized_return", 0)
    sh    = m.get("sharpe_ratio", 0)
    so    = m.get("sortino_ratio", float("nan"))
    dd    = m.get("max_drawdown", 0)
    cal   = m.get("calmar_ratio", float("nan"))
    win   = m.get("win_rate", 0)
    cost  = m.get("total_costs", 0)
    pf    = m.get("profit_factor", float("nan"))
    exc   = ret - bh.get("total_return", 0)
    sig   = "✓" if m.get("significant") else "✗"
    pval  = m.get("p_value", float("nan"))

    so_s  = f"{so:.3f}" if not (isinstance(so, float) and math.isnan(so)) else "—"
    cal_s = f"{cal:.3f}" if not (isinstance(cal, float) and math.isnan(cal)) else "—"
    pf_s  = f"{pf:.3f}" if not (isinstance(pf, float) and math.isnan(pf)) else "—"
    pv_s  = f"{pval:.4f}" if not (isinstance(pval, float) and math.isnan(pval)) else "—"

    print(f"    Return:         {ret:+.2%}  (annualized: {ann:+.2%})")
    print(f"    vs Buy-Hold:    {exc:+.2%}")
    print(f"    Sharpe:         {sh:.3f}    Sortino: {so_s}")
    print(f"    Max Drawdown:   {dd:.2%}   Calmar: {cal_s}")
    print(f"    Win Rate:       {win:.1%}    Profit Factor: {pf_s}")
    print(f"    Total Costs:    {cost:.4f}")
    print(f"    Significance:   {sig} (p={pv_s})")


def build_markdown_report(all_results: list[dict]) -> str:
    lines = [
        "# Portfolio Backtest — Summary Report",
        "",
        f"Initial capital: ${INITIAL_CAPITAL:,.0f}",
        f"Transaction cost: {COST_PER_TRADE:.1%} per dollar of turnover",
        f"Top-K grid: {', '.join(str(k) for k in TOP_K_GRID)}",
        "Long positions: top K stocks by predicted return",
        "Short positions: bottom K stocks (long-short only)",
        f"Drawdown control: reduce size 50% when portfolio drawdown > {DRAWDOWN_THRESHOLD:.0%}",
        "",
    ]

    for horizon in HORIZONS:
        label = HORIZONS[horizon]["label"]
        lines.append(f"## {label} Horizon\n")

        hr = [r for r in all_results if r["horizon"] == horizon]
        bh_ret = next((r["metrics"]["total_return"] for r in hr if r["model"] == "buy_and_hold"), 0.0)
        hr_sorted = sorted(hr, key=lambda x: x["metrics"].get("total_return", -999), reverse=True)

        lines.append("| Model | Strategy | Top K | Return | Ann. Ret | Sharpe | MaxDD | Calmar | Win% | vs B&H | Sig |")
        lines.append("|---|---|---:|---|---|---|---|---|---|---|---|")

        for r in hr_sorted:
            m   = r["metrics"]
            ret = m.get("total_return", 0)
            ann = m.get("annualized_return", 0)
            sh  = m.get("sharpe_ratio", 0)
            dd  = m.get("max_drawdown", 0)
            cal = m.get("calmar_ratio", float("nan"))
            win = m.get("win_rate", 0)
            top_k = r.get("top_k", "")
            exc = ret - bh_ret if r["model"] != "buy_and_hold" else 0.0
            sig = "✓" if m.get("significant") else "—"
            cal_s = f"{cal:.2f}" if not (isinstance(cal, float) and math.isnan(cal)) else "—"

            lines.append(
                f"| {r['model']} | {r['strategy']} "
                f"| {top_k} | {ret:+.1%} | {ann:+.1%} | {sh:.3f} | {dd:.1%} "
                f"| {cal_s} | {win:.0%} | {exc:+.1%} | {sig} |"
            )

        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Returns compound over each holding period (1-day or 5-day)",
        "- Transaction costs are applied to turnover at each rebalance",
        "- Buy-and-hold: equal-dollar stock basket bought once, no transaction costs",
        "- Significance: one-sample t-test on period returns (H0: mean = 0)",
        "- Drawdown control activates when portfolio drops > 15% from peak",
        "- Results based on out-of-sample test set only",
        "",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)

    all_results      : list[dict]            = []
    all_equity_rows  : list[pd.DataFrame]    = []
    all_trades_rows  : list[pd.DataFrame]    = []
    all_metrics_rows : list[dict]            = []

    reference_df = load_reference_data()
    bh_equity = simulate_buy_and_hold(reference_df, INITIAL_CAPITAL)
    bh_metrics = compute_metrics(bh_equity, INITIAL_CAPITAL, holding_days=1) if not bh_equity.empty else {}

    for horizon, cfg in HORIZONS.items():
        holding = cfg["holding_days"]
        label   = cfg["label"]

        print(f"\n{'='*65}")
        print(f"  {label} Horizon  (holding period = {holding} trading day(s))")
        print(f"{'='*65}")

        # ── Discover all prediction files ──────────────────────────────────────
        model_paths = discover_models(horizon)
        if not model_paths:
            print(f"  [WARN] No prediction files found for horizon {horizon}")
            continue

        print(f"\n  Found {len(model_paths)} models:")
        loaded: dict[str, pd.DataFrame] = {}
        for model_name, path in sorted(model_paths.items()):
            df = load_predictions(model_name, path)
            if df is not None:
                loaded[model_name] = df

        if not loaded:
            print("  [WARN] No predictions could be loaded")
            continue

        # ── Buy-and-Hold benchmark ─────────────────────────────────────────────
        bh_eq = bh_equity.copy()
        bh_m = bh_metrics.copy()

        bh_eq["model"]    = "buy_and_hold"
        bh_eq["strategy"] = "buy_and_hold"
        bh_eq["horizon"]  = horizon
        bh_eq["top_k"]    = 0
        all_equity_rows.append(bh_eq)
        all_results.append({"model": "buy_and_hold", "strategy": "buy_and_hold",
                             "horizon": horizon, "top_k": 0, "metrics": bh_m})
        bh_row = {"horizon": horizon, "model": "buy_and_hold", "strategy": "buy_and_hold", "top_k": 0}
        bh_row.update(bh_m)
        bh_row["excess_vs_bh"] = 0.0
        all_metrics_rows.append(bh_row)

        print(f"\n  Buy-and-Hold: {bh_m.get('total_return', 0):+.2%}  "
              f"Sharpe: {bh_m.get('sharpe_ratio', 0):.3f}")

        # ── Each model and strategy ────────────────────────────────────────────
        for model_name, df in sorted(loaded.items()):
            if model_name == "buy_and_hold":
                continue

            print(f"\n  ── {model_name} ──")

            for top_k in TOP_K_GRID:
                if len(df["ticker"].unique()) < top_k * 2:
                    print(f"\n    [top_k={top_k}] skipped; not enough tickers")
                    continue

                for strategy in ("long_only", "long_short"):
                    print(f"\n    [{strategy} | top_k={top_k}]")

                    equity_df, trades_df = simulate_portfolio(
                        df, strategy, holding, top_k, INITIAL_CAPITAL,
                    )

                    if equity_df.empty:
                        print("    [SKIP] No equity data")
                        continue

                    m = compute_metrics(equity_df, INITIAL_CAPITAL, holding_days=holding)
                    print_metrics(model_name, strategy, m, bh_m)

                    # Tag dataframes
                    equity_df["model"]    = model_name
                    equity_df["strategy"] = strategy
                    equity_df["horizon"]  = horizon
                    equity_df["top_k"]    = top_k
                    trades_df["model"]    = model_name
                    trades_df["strategy"] = strategy
                    trades_df["horizon"]  = horizon
                    trades_df["top_k"]    = top_k

                    all_equity_rows.append(equity_df)
                    all_trades_rows.append(trades_df)

                    all_results.append({
                        "model":    model_name,
                        "strategy": strategy,
                        "horizon":  horizon,
                        "top_k":    top_k,
                        "metrics":  m,
                    })

                    row = {"horizon": horizon, "model": model_name, "strategy": strategy, "top_k": top_k}
                    row.update(m)
                    row["excess_vs_bh"] = m.get("total_return", 0) - bh_m.get("total_return", 0)
                    all_metrics_rows.append(row)

    # ── Save outputs ───────────────────────────────────────────────────────────

    if all_equity_rows:
        equity_all = pd.concat(all_equity_rows, ignore_index=True)
        for h in HORIZONS:
            sub = equity_all[equity_all["horizon"] == h]
            if not sub.empty:
                sub.to_csv(REPORT_DIR / f"equity_curves_{h}.csv", index=False)

    if all_trades_rows:
        trades_all = pd.concat(all_trades_rows, ignore_index=True)
        for h in HORIZONS:
            sub = trades_all[trades_all["horizon"] == h]
            if not sub.empty:
                sub.to_csv(REPORT_DIR / f"trades_{h}.csv", index=False)

    if all_metrics_rows:
        mdf = pd.DataFrame(all_metrics_rows)
        for h in HORIZONS:
            sub = mdf[mdf["horizon"] == h].sort_values("total_return", ascending=False)
            if not sub.empty:
                sub.to_csv(REPORT_DIR / f"metrics_{h}.csv", index=False)

    report_path = MD_DIR / "backtest_report.md"
    report_path.write_text(build_markdown_report(all_results), encoding="utf-8")

    # ── Final summary ──────────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print("  BACKTEST COMPLETE")
    print(f"{'='*65}")

    if all_metrics_rows:
        mdf = pd.DataFrame(all_metrics_rows)
        for h in HORIZONS:
            sub = mdf[mdf["horizon"] == h].sort_values("total_return", ascending=False)
            if sub.empty:
                continue
            lbl = HORIZONS[h]["label"]
            print(f"\n  {lbl} — All Results (sorted by return, after transaction costs):")
            print(f"  {'Model':<28} {'Strategy':<12} {'TopK':>4} {'Return':>9} {'Ann.':>8} "
                  f"{'Sharpe':>7} {'MaxDD':>8} {'vs B&H':>9}")
            print(f"  {'─'*28} {'─'*12} {'─'*4} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*9}")
            for _, r in sub.iterrows():
                ann = r.get("annualized_return", float("nan"))
                ann_s = f"{ann:>+7.1%}" if not (isinstance(ann, float) and math.isnan(ann)) else "      —"
                top_k = "" if r.get("top_k", 0) == 0 else int(r["top_k"])
                print(
                    f"  {r['model']:<28} {r['strategy']:<12} "
                    f"{str(top_k):>4} "
                    f"{r['total_return']:>+8.1%} {ann_s} "
                    f"{r['sharpe_ratio']:>7.3f} {r['max_drawdown']:>7.1%} "
                    f"{r['excess_vs_bh']:>+8.1%}"
                )

    print(f"\n  Output: {REPORT_DIR}")
    for f in sorted(REPORT_DIR.iterdir()):
        if f.is_file():
            print(f"    {f.name:<35} {f.stat().st_size:>8,} bytes")
        elif f.is_dir():
            for sub in sorted(f.iterdir()):
                print(f"    {f.name}/{sub.name:<30} {sub.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    main()
