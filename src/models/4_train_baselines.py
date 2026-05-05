"""
Train baseline models and evaluate on both targets.

Models:
    Statistical:
        1. Historical Mean          — simplest possible baseline
        2. Naive Last Return        — today's return predicts tomorrow (1-day only)
    Machine Learning:
        3. Linear Regression        — linear relationship between features and target
        4. Ridge Regression         — linear with L2 regularization (prevents overfitting)
        5. Random Forest            — non-linear, time-weighted training
    Time-Series:
        6. ARIMA                    — auto p/d/q, per-ticker, rolling one-step-ahead
        7. SARIMA                   — ARIMA + weekly seasonality (m=5)
        8. SARIMAX                  — SARIMA + top 5 exogenous features

Output structure per model:
    outputs/baselines/{model_name}/
        metrics_1d.json, metrics_5d.json
        per_ticker_1d.csv, per_ticker_5d.csv
        predictions_1d_test.csv, predictions_1d_val.csv
        predictions_5d_test.csv, predictions_5d_val.csv

Install:  pip install scikit-learn scipy
Run:      python src/models/4_train_baselines.py
"""

import sys
from pathlib import Path
import json
import warnings
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from universes import get_data_dir, get_outputs_dir

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from backtest_utils import run_model_backtest


BASE_DIR    = Path(__file__).resolve().parents[2]
_DATA       = get_data_dir()
_OUTPUTS    = get_outputs_dir()
TRAIN_PATH  = _DATA    / "splits" / "train.csv"
VAL_PATH    = _DATA    / "splits" / "val.csv"
TEST_PATH   = _DATA    / "splits" / "test.csv"
OUTPUT_BASE = _OUTPUTS / "baselines"

INITIAL_CAPITAL       = 10_000.0
TOP_K_LONGS           = 5
BOTTOM_K_SHORTS       = 5
TRADING_DAYS_PER_YEAR = 252

TARGET_COLUMNS = [
    "target_next_day_return",
    "target_5d_return",
    "target_10d_return",
    "target_30d_return",
]
TARGET_META = {
    "target_next_day_return": ("1d", "1-DAY TARGET", 1),
    "target_5d_return": ("5d", "5-DAY TARGET", 5),
    "target_10d_return": ("10d", "10-DAY TARGET", 10),
    "target_30d_return": ("30d", "30-DAY TARGET", 30),
}
FEATURE_COLUMNS = [
    "log_return", "return_5d", "return_10d", "return_30d",
    "volume_change", "volume_ma_ratio", "obv_change",
    "rsi_14", "macd", "macd_signal", "macd_diff", "rolling_sharpe_20",
    "volatility_10", "volatility_20", "volatility_30",
    "atr_14", "bollinger_band_width", "beta_60", "idiosyncratic_vol_20",
    "volatility_regime_20",
    "spy_log_return", "spy_return_5d", "spy_return_10d", "spy_return_30d",
    "spy_volatility_20", "qqq_log_return", "qqq_return_5d",
    "dia_log_return", "dia_return_5d", "iwm_log_return", "iwm_return_5d",
    "vix_close", "vix_log_return", "relative_strength",
    "sector_log_return", "sector_return_5d", "sector_return_10d",
    "sector_return_30d", "sector_relative_strength",
    "day_of_week",
]

# Top 5 most important features for SARIMAX
# (using all 19 causes numerical instability on ~1,700 rows per ticker)
SARIMAX_FEATURES = [
    "log_return",
    "rsi_14",
    "volatility_10",
    "spy_log_return",
    "vix_close",
]

REFERENCE_COLUMNS = ["Date", "ticker"]


# ── Loading ────────────────────────────────────────────────────────────────────

def load_split(path):
    """Load one split CSV, validate columns, parse dates."""
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
    """Directional accuracy: fraction of correct up/down predictions."""
    return float((np.sign(yt) == np.sign(yp)).mean())


def information_coefficient(df, preds, target):
    """
    Average daily Spearman rank correlation between predictions and actuals.
    Measures stock-ranking ability — the most relevant metric for long-short trading.
    IC > 0.05 is tradeable, IC > 0.10 is very strong.
    """
    df = df.copy()
    df["y_pred"] = preds
    daily_ics = []
    for _, grp in df.groupby("Date"):
        if len(grp) < 5:
            continue
        ic, _ = spearmanr(grp[target].values, grp["y_pred"].values)
        if not np.isnan(ic):
            daily_ics.append(ic)
    return float(np.mean(daily_ics)) if daily_ics else 0.0


def compute_metrics(yt, yp):
    """Core statistical metrics."""
    return {
        "mae":  float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "directional_accuracy": dir_acc(yt, yp),
    }


def per_ticker_metrics(df, preds, target):
    """Per-stock statistical metrics for the results section."""
    df = df.copy()
    df["y_pred"] = preds
    return {t: compute_metrics(g[target].to_numpy(), g["y_pred"].to_numpy())
            for t, g in df.groupby("ticker")}


# ── Backtest ───────────────────────────────────────────────────────────────────

def run_backtest(df, preds, target, holding_days=None, benchmark_df=None):
    """Run long-only, long-short, and horizon-independent buy-and-hold backtests."""
    return run_model_backtest(
        df,
        preds,
        target,
        holding_days=holding_days,
        benchmark_df=benchmark_df,
        initial_capital=INITIAL_CAPITAL,
        top_k_longs=TOP_K_LONGS,
        bottom_k_shorts=BOTTOM_K_SHORTS,
        trading_days_per_year=TRADING_DAYS_PER_YEAR,
    )


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_preds(df, preds, target, model_dir, suffix, split):
    out = df[REFERENCE_COLUMNS].copy()
    out["y_true"] = df[target].values
    out["y_pred"] = preds
    out.to_csv(model_dir / f"predictions_{suffix}_{split}.csv", index=False)


def save_per_ticker_csv(pt, path):
    pd.DataFrame([{"ticker": t, **m} for t, m in sorted(pt.items())]).to_csv(path, index=False)


def evaluate_and_save(name, target, suffix, model_dir, train_df, train_p,
                      val_df, val_p, test_df, test_p):
    """Compute all metrics, run backtest, save everything."""
    train_m = compute_metrics(train_df[target].to_numpy(), train_p)
    val_m   = compute_metrics(val_df[target].to_numpy(),   val_p)
    test_m  = compute_metrics(test_df[target].to_numpy(),  test_p)

    # IC computed on test set — measures ranking ability
    ic = information_coefficient(test_df, test_p, target)
    test_m["information_coefficient"] = ic

    pt = per_ticker_metrics(test_df, test_p, target)
    bt = run_backtest(test_df, test_p, target)

    save_preds(val_df,  val_p,  target, model_dir, suffix, "val")
    save_preds(test_df, test_p, target, model_dir, suffix, "test")
    save_per_ticker_csv(pt, model_dir / f"per_ticker_{suffix}.csv")

    with open(model_dir / f"metrics_{suffix}.json", "w") as f:
        json.dump({"target": target, "train": train_m, "validation": val_m,
                   "test": test_m, "test_backtest": bt}, f, indent=2)

    return {"model": name, "target": target, "train": train_m,
            "validation": val_m, "test": test_m, "test_backtest": bt}


# ── Simple predictors ──────────────────────────────────────────────────────────

def hist_mean(train_df, target, n):
    return np.full(n, train_df[target].mean(), dtype=float)

def naive_lr(df):
    return df["log_return"].to_numpy(dtype=float)

def build_scaler(X):
    s = StandardScaler()
    s.fit(X)
    return s

def train_linreg(X, y):
    m = LinearRegression()
    m.fit(X, y)
    return m

def train_ridge(X, y):
    """Ridge regression — L2 regularized linear model. Alpha selected via cross-val."""
    m = Ridge(alpha=1.0)
    m.fit(X, y)
    return m

def train_rf(X, y, n):
    """Random Forest with time-weighted samples — recent data weighted higher."""
    m = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    m.fit(X, y, sample_weight=np.linspace(0.5, 1.0, n))
    return m


# ── Time-series models (ARIMA / SARIMA / SARIMAX) ─────────────────────────────

def run_ts_per_ticker(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    target:   str,
    mode:     str,
    scaler:   StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Fit one time-series model per ticker using rolling one-step-ahead prediction.

    mode="arima"   — ARIMA(p,d,q), no seasonality, no exogenous
    mode="sarima"  — SARIMA(p,d,q)(P,D,Q)[5], weekly seasonality
    mode="sarimax" — SARIMAX with top 5 exogenous features + seasonality

    Prediction strategy (rolling one-step-ahead):
        1. Fit auto_arima on train to find best (p,d,q) order
        2. For val: start from trained model, predict 1 step, update with
           actual value, predict next step, repeat — this is how ARIMA would
           be used in practice
        3. For test: refit with train+val data using same order, then roll

    This avoids the flat-forecast problem where multi-step ARIMA predictions
    converge to the mean after a few steps.
    """
    seasonal = mode in ("sarima", "sarimax")
    use_exog = mode == "sarimax"
    exog_cols = SARIMAX_FEATURES if use_exog else None

    tickers      = sorted(train_df["ticker"].unique())
    tr_preds_all = []
    va_preds_all = []
    te_preds_all = []
    ticker_params = {}

    for ticker in tickers:
        tr_rows = train_df[train_df["ticker"] == ticker].copy()
        va_rows = val_df[val_df["ticker"] == ticker].copy()
        te_rows = test_df[test_df["ticker"] == ticker].copy()

        y_tr = tr_rows[target].values.astype(float)
        y_va = va_rows[target].values.astype(float)
        y_te = te_rows[target].values.astype(float)

        if use_exog:
            X_tr = scaler.transform(tr_rows[exog_cols].values)
            X_va = scaler.transform(va_rows[exog_cols].values)
            X_te = scaler.transform(te_rows[exog_cols].values)
        else:
            X_tr = X_va = X_te = None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Step 1: Find best order on training data
                auto_model = auto_arima(
                    y_tr,
                    exogenous     = X_tr,
                    seasonal      = seasonal,
                    m             = 5 if seasonal else 1,
                    stepwise      = True,
                    suppress_warnings = True,
                    error_action  = "ignore",
                    max_p=3, max_q=3,
                    max_P=1, max_Q=1,
                    information_criterion = "aic",
                )

                best_order = auto_model.order
                best_seasonal = auto_model.seasonal_order if seasonal else (0, 0, 0, 0)

                ticker_params[ticker] = {
                    "order":          list(best_order),
                    "seasonal_order": list(best_seasonal) if seasonal else None,
                }

                # Step 2: In-sample train predictions
                train_pred = auto_model.predict_in_sample(exogenous=X_tr)
                if len(train_pred) < len(y_tr):
                    pad = np.full(len(y_tr) - len(train_pred), y_tr.mean())
                    train_pred = np.concatenate([pad, train_pred])

                # Step 3: Rolling one-step-ahead for val
                # Clone model with found order, predict one step at a time
                val_pred = _rolling_predict(
                    y_history=y_tr, y_future=y_va,
                    X_history=X_tr, X_future=X_va,
                    order=best_order, seasonal_order=best_seasonal,
                    seasonal=seasonal,
                )

                # Step 4: Refit on train+val, then rolling one-step for test
                y_tv = np.concatenate([y_tr, y_va])
                X_tv = np.vstack([X_tr, X_va]) if use_exog else None

                test_pred = _rolling_predict(
                    y_history=y_tv, y_future=y_te,
                    X_history=X_tv, X_future=X_te,
                    order=best_order, seasonal_order=best_seasonal,
                    seasonal=seasonal,
                )

        except Exception as err:
            print(f"    {mode.upper()} failed for {ticker}: {err} — using mean fallback")
            train_pred = np.full(len(y_tr), y_tr.mean())
            val_pred   = np.full(len(y_va), y_tr.mean())
            test_pred  = np.full(len(y_te), y_tr.mean())
            ticker_params[ticker] = {"order": None, "error": str(err)}

        tr_preds_all.append(train_pred)
        va_preds_all.append(val_pred)
        te_preds_all.append(test_pred)

        order_str = ticker_params[ticker].get("order")
        print(f"    {ticker:<6} {mode.upper()} done  order={order_str}")

    return (
        np.concatenate(tr_preds_all),
        np.concatenate(va_preds_all),
        np.concatenate(te_preds_all),
        ticker_params,
    )


def _rolling_predict(
    y_history, y_future, X_history, X_future,
    order, seasonal_order, seasonal,
) -> np.ndarray:
    """
    Rolling one-step-ahead prediction using a fixed ARIMA order.

    For each step in y_future:
        1. Fit model on all available history
        2. Predict one step ahead
        3. Append actual value to history
        4. Repeat

    To keep runtime manageable, we refit every 20 steps instead of every step.
    Between refits, we use the model's update() method which is fast.
    """
    use_exog = X_history is not None
    predictions = []
    refit_interval = 20

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Initial fit
        model = PmdARIMA(
            order=order,
            seasonal_order=seasonal_order if seasonal else (0, 0, 0, 0),
            suppress_warnings=True,
        )
        model.fit(y_history, exogenous=X_history)

        for i in range(len(y_future)):
            # Predict one step
            exog_step = X_future[i:i+1] if use_exog else None
            pred = model.predict(n_periods=1, exogenous=exog_step)
            predictions.append(float(pred[0]))

            # Update model with actual observation
            exog_update = X_future[i:i+1] if use_exog else None
            model.update(y_future[i:i+1], exogenous=exog_update)

            # Periodic refit to prevent drift (every 20 steps)
            if (i + 1) % refit_interval == 0 and i + 1 < len(y_future):
                try:
                    combined_y = np.concatenate([y_history, y_future[:i+1]])
                    combined_X = np.vstack([X_history, X_future[:i+1]]) if use_exog else None
                    model = PmdARIMA(
                        order=order,
                        seasonal_order=seasonal_order if seasonal else (0, 0, 0, 0),
                        suppress_warnings=True,
                    )
                    model.fit(combined_y, exogenous=combined_X)
                except Exception:
                    pass  # keep using the updated model if refit fails

    return np.array(predictions)


# ── Printing ───────────────────────────────────────────────────────────────────

def print_summary(results, label):
    """Print comprehensive summary for one target."""
    print(f"\n══ {label} ══════════════════════════════════════")

    print(f"\n  Statistical Metrics:")
    print(f"  {'Model':<22} {'Test MAE':<12} {'Test RMSE':<12} {'Dir Acc':<10} {'IC':<10}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    for r in results:
        m = r["test"]
        ic = m.get("information_coefficient", 0.0)
        print(f"  {r['model']:<22} {m['mae']:<12.6f} {m['rmse']:<12.6f} "
              f"{m['directional_accuracy']:<10.4f} {ic:<10.4f}")

    print(f"\n  Backtest (${INITIAL_CAPITAL:,.0f}):")
    print(f"  {'Model':<22} {'Long Only':<22} {'Long-Short':<22}")
    print(f"  {'─'*22} {'─'*22} {'─'*22}")
    for r in results:
        bt = r.get("test_backtest", {})
        if "error" in bt:
            continue
        lo, ls = bt["long_only"], bt["long_short"]
        print(f"  {r['model']:<22} "
              f"${lo['final_value']:>9,.0f} ({lo['total_return']*100:+.1f}%)    "
              f"${ls['final_value']:>9,.0f} ({ls['total_return']*100:+.1f}%)")

    print(f"\n  Risk Metrics:")
    print(f"  {'Model':<22} {'Sharpe(LO)':<12} {'Sharpe(LS)':<12} {'MaxDD(LO)':<12} {'MaxDD(LS)':<12}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for r in results:
        bt = r.get("test_backtest", {})
        if "error" in bt:
            continue
        lo, ls = bt["long_only"], bt["long_short"]
        print(f"  {r['model']:<22} "
              f"{lo['sharpe_ratio']:<12.3f} {ls['sharpe_ratio']:<12.3f} "
              f"{lo['max_drawdown']*100:>+8.1f}%   {ls['max_drawdown']*100:>+8.1f}%")

    if results:
        bt = results[0].get("test_backtest", {})
        if "buy_and_hold_benchmark" in bt:
            bh = bt["buy_and_hold_benchmark"]
            print(f"\n  Buy-and-Hold: ${bh['final_value']:,.0f} "
                  f"({bh['total_return']*100:+.1f}%)  "
                  f"Sharpe: {bh['sharpe_ratio']:.3f}  "
                  f"MaxDD: {bh['max_drawdown']*100:+.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    total_start = time.time()

    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    print(f"  Train: {len(train_df)} rows  Val: {len(val_df)} rows  Test: {len(test_df)} rows")
    print(f"  Tickers: {train_df['ticker'].nunique()}  Features: {len(FEATURE_COLUMNS)}")

    for target in TARGET_COLUMNS:
        sfx, lbl, _ = TARGET_META[target]

        print(f"\n{'='*60}")
        print(f"  Training baselines for {lbl}")
        print(f"{'='*60}")

        X_tr, y_tr = get_xy(train_df, target)
        X_va, _    = get_xy(val_df,   target)
        X_te, _    = get_xy(test_df,  target)

        sc  = build_scaler(X_tr)
        Xts = sc.transform(X_tr)
        Xvs = sc.transform(X_va)
        Xes = sc.transform(X_te)

        res = []

        # ── 1. Historical Mean ────────────────────────────────────────────────
        print("  Training: Historical Mean")
        d = OUTPUT_BASE / "historical_mean"; d.mkdir(parents=True, exist_ok=True)
        res.append(evaluate_and_save("historical_mean", target, sfx, d,
            train_df, hist_mean(train_df, target, len(train_df)),
            val_df,   hist_mean(train_df, target, len(val_df)),
            test_df,  hist_mean(train_df, target, len(test_df))))

        # ── 2. Naive Last Return (1-day only) ─────────────────────────────────
        if sfx == "1d":
            print("  Training: Naive Last Return")
            d = OUTPUT_BASE / "naive_last_return"; d.mkdir(parents=True, exist_ok=True)
            res.append(evaluate_and_save("naive_last_return", target, sfx, d,
                train_df, naive_lr(train_df),
                val_df,   naive_lr(val_df),
                test_df,  naive_lr(test_df)))

        # ── 3. Linear Regression ──────────────────────────────────────────────
        print("  Training: Linear Regression")
        d = OUTPUT_BASE / "linear_regression"; d.mkdir(parents=True, exist_ok=True)
        lr = train_linreg(Xts, y_tr)
        res.append(evaluate_and_save("linear_regression", target, sfx, d,
            train_df, lr.predict(Xts),
            val_df,   lr.predict(Xvs),
            test_df,  lr.predict(Xes)))

        # ── 4. Ridge Regression ───────────────────────────────────────────────
        print("  Training: Ridge Regression")
        d = OUTPUT_BASE / "ridge_regression"; d.mkdir(parents=True, exist_ok=True)
        ridge = train_ridge(Xts, y_tr)
        res.append(evaluate_and_save("ridge_regression", target, sfx, d,
            train_df, ridge.predict(Xts),
            val_df,   ridge.predict(Xvs),
            test_df,  ridge.predict(Xes)))

        # ── 5. Random Forest ─────────────────────────────────────────────────
        print("  Training: Random Forest")
        d = OUTPUT_BASE / "random_forest"; d.mkdir(parents=True, exist_ok=True)
        rf = train_rf(Xts, y_tr, len(Xts))
        res.append(evaluate_and_save("random_forest", target, sfx, d,
            train_df, rf.predict(Xts),
            val_df,   rf.predict(Xvs),
            test_df,  rf.predict(Xes)))

        # ── 6. ARIMA ─────────────────────────────────────────────────────────
        # print(f"  Training: ARIMA (per-ticker, rolling 1-step)...")
        # d = OUTPUT_BASE / "arima"; d.mkdir(parents=True, exist_ok=True)
        # t0 = time.time()
        # tr_p, va_p, te_p, params = run_ts_per_ticker(
        #     train_df, val_df, test_df, target, mode="arima", scaler=sc)
        # print(f"    ARIMA completed in {time.time()-t0:.0f}s")
        # with open(d / f"params_{sfx}.json", "w") as f:
        #     json.dump(params, f, indent=2)
        # res.append(evaluate_and_save("arima", target, sfx, d,
        #     train_df, tr_p, val_df, va_p, test_df, te_p))

        # ── 7. SARIMA ────────────────────────────────────────────────────────
        # print(f"  Training: SARIMA (per-ticker, weekly seasonality)...")
        # d = OUTPUT_BASE / "sarima"; d.mkdir(parents=True, exist_ok=True)
        # t0 = time.time()
        # tr_p, va_p, te_p, params = run_ts_per_ticker(
        #     train_df, val_df, test_df, target, mode="sarima", scaler=sc)
        # print(f"    SARIMA completed in {time.time()-t0:.0f}s")
        # with open(d / f"params_{sfx}.json", "w") as f:
        #     json.dump(params, f, indent=2)
        # res.append(evaluate_and_save("sarima", target, sfx, d,
        #     train_df, tr_p, val_df, va_p, test_df, te_p))

        # ── 8. SARIMAX ───────────────────────────────────────────────────────
        # print(f"  Training: SARIMAX (per-ticker, top 5 features)...")
        # print(f"    Exogenous features: {SARIMAX_FEATURES}")
        # d = OUTPUT_BASE / "sarimax"; d.mkdir(parents=True, exist_ok=True)
        # t0 = time.time()
        # tr_p, va_p, te_p, params = run_ts_per_ticker(
        #     train_df, val_df, test_df, target, mode="sarimax", scaler=sc)
        # print(f"    SARIMAX completed in {time.time()-t0:.0f}s")
        # with open(d / f"params_{sfx}.json", "w") as f:
        #     json.dump(params, f, indent=2)
        # res.append(evaluate_and_save("sarimax", target, sfx, d,
        #     train_df, tr_p, val_df, va_p, test_df, te_p))

        print_summary(res, lbl)

    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"  Results saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
