"""
Train baseline models and evaluate on both targets.

Output structure:
    outputs/baselines/{model_name}/
        metrics_1d.json          — aggregate stats + backtest for 1-day
        metrics_5d.json          — aggregate stats + backtest for 5-day
        per_ticker_1d.csv        — per-stock metrics for 1-day
        per_ticker_5d.csv        — per-stock metrics for 5-day
        predictions_1d_test.csv  — test set predictions
        predictions_1d_val.csv   — val set predictions
        predictions_5d_test.csv
        predictions_5d_val.csv

Run:  python src/models/4_train_baselines.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


BASE_DIR    = Path(__file__).resolve().parents[2]
TRAIN_PATH  = BASE_DIR / "data" / "splits" / "train.csv"
VAL_PATH    = BASE_DIR / "data" / "splits" / "val.csv"
TEST_PATH   = BASE_DIR / "data" / "splits" / "test.csv"
OUTPUT_BASE = BASE_DIR / "outputs" / "baselines"

INITIAL_CAPITAL       = 10_000.0
TOP_K_LONGS           = 5
BOTTOM_K_SHORTS       = 5
TRADING_DAYS_PER_YEAR = 252

TARGET_COLUMNS = ["target_next_day_return", "target_5d_return"]
FEATURE_COLUMNS = [
    "log_return", "return_5d", "return_10d",
    "volume_change", "volume_ma_ratio", "obv_change",
    "rsi_14", "macd", "macd_signal", "macd_diff", "rolling_sharpe_20",
    "volatility_10", "atr_14", "bollinger_band_width",
    "spy_log_return", "vix_close", "vix_log_return", "relative_strength",
    "day_of_week",
]
REFERENCE_COLUMNS = ["Date", "ticker"]


# ── Loading ────────────────────────────────────────────────────────────────────

def load_split(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    required = REFERENCE_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


def get_xy(df, target):
    return df[FEATURE_COLUMNS].copy(), df[target].copy()


# ── Metrics ────────────────────────────────────────────────────────────────────

def dir_acc(yt, yp):
    return float((np.sign(yt) == np.sign(yp)).mean())

def compute_metrics(yt, yp):
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "directional_accuracy": dir_acc(yt, yp),
    }

def per_ticker_metrics(df, preds, target):
    df = df.copy(); df["y_pred"] = preds
    out = {}
    for t, g in df.groupby("ticker"):
        out[t] = compute_metrics(g[target].to_numpy(), g["y_pred"].to_numpy())
    return out


# ── Backtest ───────────────────────────────────────────────────────────────────

def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(r) > 0 and r.std() > 0 else 0.0

def max_dd(eq):
    pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min()) if len(eq) > 0 else 0.0

def run_backtest(df, preds, target):
    df = df.copy(); df["y_pred"] = preds; df["y_true"] = df[target]
    lo, ls, bh = [], [], []
    for _, g in df.groupby("Date"):
        if len(g) < TOP_K_LONGS + BOTTOM_K_SHORTS: continue
        s = g.sort_values("y_pred", ascending=False)
        t = s.head(TOP_K_LONGS)["y_true"].mean()
        b = s.tail(BOTTOM_K_SHORTS)["y_true"].mean()
        lo.append(t); ls.append((t - b) / 2.0); bh.append(g["y_true"].mean())
    if not lo: return {"error": "Not enough data"}
    def _b(r):
        si = np.exp(np.array(r)) - 1; eq = INITIAL_CAPITAL * np.cumprod(1 + si)
        return {"final_value": float(eq[-1]), "total_return": float(eq[-1]/INITIAL_CAPITAL-1),
                "sharpe_ratio": sharpe(si), "max_drawdown": max_dd(eq), "n_trading_days": len(si)}
    return {"long_only": _b(lo), "long_short": _b(ls), "buy_and_hold_benchmark": _b(bh)}


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_preds(df, preds, target, model_dir, suffix, split):
    out = df[REFERENCE_COLUMNS].copy()
    out["y_true"] = df[target].values; out["y_pred"] = preds
    out.to_csv(model_dir / f"predictions_{suffix}_{split}.csv", index=False)

def save_per_ticker_csv(pt, path):
    rows = [{"ticker": t, **m} for t, m in sorted(pt.items())]
    pd.DataFrame(rows).to_csv(path, index=False)

def evaluate_and_save(name, target, suffix, model_dir, train_df, train_p, val_df, val_p, test_df, test_p):
    train_m = compute_metrics(train_df[target].to_numpy(), train_p)
    val_m   = compute_metrics(val_df[target].to_numpy(), val_p)
    test_m  = compute_metrics(test_df[target].to_numpy(), test_p)
    pt = per_ticker_metrics(test_df, test_p, target)
    bt = run_backtest(test_df, test_p, target)

    save_preds(val_df,  val_p,  target, model_dir, suffix, "val")
    save_preds(test_df, test_p, target, model_dir, suffix, "test")
    save_per_ticker_csv(pt, model_dir / f"per_ticker_{suffix}.csv")

    results = {"target": target, "train": train_m, "validation": val_m, "test": test_m, "test_backtest": bt}
    with open(model_dir / f"metrics_{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)

    return {"model": name, "target": target, "train": train_m, "validation": val_m,
            "test": test_m, "test_backtest": bt}


# ── Predictors ─────────────────────────────────────────────────────────────────

def hist_mean(train_df, target, n): return np.full(n, train_df[target].mean(), dtype=float)
def naive_lr(df): return df["log_return"].to_numpy(dtype=float)
def build_scaler(X):
    s = StandardScaler(); s.fit(X); return s
def train_lr(X, y):
    m = LinearRegression(); m.fit(X, y); return m
def train_rf(X, y, n):
    m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    m.fit(X, y, sample_weight=np.linspace(0.5, 1.0, n)); return m


# ── Printing ───────────────────────────────────────────────────────────────────

def print_summary(results, label):
    print(f"\n══ {label} ══════════════════════════════════════")
    print(f"\n  Statistical Metrics:")
    for r in results:
        print(f"  {r['model']}")
        for s in ("train", "validation", "test"):
            m = r[s]
            print(f"    {s.capitalize():<12} -> MAE: {m['mae']:.6f}  RMSE: {m['rmse']:.6f}  Dir Acc: {m['directional_accuracy']:.4f}")

    print(f"\n  Backtest (${INITIAL_CAPITAL:,.0f}):")
    print(f"  {'Model':<22} {'Long Only':<22} {'Long-Short':<22}")
    print(f"  {'─'*22} {'─'*22} {'─'*22}")
    for r in results:
        bt = r.get("test_backtest", {}); 
        if "error" in bt: continue
        lo, ls = bt["long_only"], bt["long_short"]
        print(f"  {r['model']:<22} ${lo['final_value']:>9,.0f} ({lo['total_return']*100:+.1f}%)    ${ls['final_value']:>9,.0f} ({ls['total_return']*100:+.1f}%)")

    print(f"\n  Risk Metrics:")
    print(f"  {'Model':<22} {'Sharpe(LO)':<12} {'Sharpe(LS)':<12} {'MaxDD(LO)':<12} {'MaxDD(LS)':<12}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for r in results:
        bt = r.get("test_backtest", {}); 
        if "error" in bt: continue
        lo, ls = bt["long_only"], bt["long_short"]
        print(f"  {r['model']:<22} {lo['sharpe_ratio']:<12.3f} {ls['sharpe_ratio']:<12.3f} {lo['max_drawdown']*100:>+8.1f}%   {ls['max_drawdown']*100:>+8.1f}%")

    if results:
        bt = results[0].get("test_backtest", {})
        if "buy_and_hold_benchmark" in bt:
            bh = bt["buy_and_hold_benchmark"]
            print(f"\n  Buy-and-Hold: ${bh['final_value']:,.0f} ({bh['total_return']*100:+.1f}%)  Sharpe: {bh['sharpe_ratio']:.3f}  MaxDD: {bh['max_drawdown']*100:+.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    for target in TARGET_COLUMNS:
        sfx = "1d" if "next_day" in target else "5d"
        lbl = "1-DAY TARGET" if sfx == "1d" else "5-DAY TARGET"

        X_tr, y_tr = get_xy(train_df, target)
        X_va, y_va = get_xy(val_df, target)
        X_te, y_te = get_xy(test_df, target)
        sc = build_scaler(X_tr)
        Xts, Xvs, Xes = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)

        res = []

        d = OUTPUT_BASE / "historical_mean"; d.mkdir(parents=True, exist_ok=True)
        res.append(evaluate_and_save("historical_mean", target, sfx, d,
            train_df, hist_mean(train_df,target,len(train_df)),
            val_df, hist_mean(train_df,target,len(val_df)),
            test_df, hist_mean(train_df,target,len(test_df))))

        if sfx == "1d":
            d = OUTPUT_BASE / "naive_last_return"; d.mkdir(parents=True, exist_ok=True)
            res.append(evaluate_and_save("naive_last_return", target, sfx, d,
                train_df, naive_lr(train_df), val_df, naive_lr(val_df), test_df, naive_lr(test_df)))

        d = OUTPUT_BASE / "linear_regression"; d.mkdir(parents=True, exist_ok=True)
        lr = train_lr(Xts, y_tr)
        res.append(evaluate_and_save("linear_regression", target, sfx, d,
            train_df, lr.predict(Xts), val_df, lr.predict(Xvs), test_df, lr.predict(Xes)))

        d = OUTPUT_BASE / "random_forest"; d.mkdir(parents=True, exist_ok=True)
        rf = train_rf(Xts, y_tr, len(Xts))
        res.append(evaluate_and_save("random_forest", target, sfx, d,
            train_df, rf.predict(Xts), val_df, rf.predict(Xvs), test_df, rf.predict(Xes)))

        print_summary(res, lbl)

    print(f"\n  Results saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
