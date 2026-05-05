"""
Train LSTM model with BOTH default and tuned configurations.

Output structure:
    outputs/lstm/default/
        config.json
        metrics_1d.json, metrics_5d.json
        per_ticker_1d.csv, per_ticker_5d.csv
        predictions_1d_test.csv, predictions_5d_test.csv
        model_1d.pt, model_5d.pt
    outputs/lstm/tuned/     (only if lstm_best_config.json exists)
        same structure

    outputs/shared/
        scaler.pkl, ticker_map.json

Run:  python src/models/5_train_lstm.py
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from universes import get_data_dir, get_outputs_dir

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import joblib

from backtest_utils import run_model_backtest


BASE_DIR    = Path(__file__).resolve().parents[2]
_DATA       = get_data_dir()
_OUTPUTS    = get_outputs_dir()
TRAIN_PATH  = _DATA    / "splits" / "train.csv"
VAL_PATH    = _DATA    / "splits" / "val.csv"
TEST_PATH   = _DATA    / "splits" / "test.csv"
OUTPUT_BASE = _OUTPUTS / "lstm"
SHARED_DIR  = _OUTPUTS / "shared"
TUNED_CFG   = _OUTPUTS / "shared" / "lstm_best_config.json"

RANDOM_SEED = 42
INITIAL_CAPITAL       = 10_000.0
TOP_K_LONGS           = 5
BOTTOM_K_SHORTS       = 5
TRADING_DAYS_PER_YEAR = 252

DEFAULT_CONFIG = {
    "lookback": 20, "hidden_size": 128, "num_layers": 2, "dropout": 0.2,
    "batch_size": 64, "learning_rate": 5e-4, "max_epochs": 50, "patience": 10,
    "embedding_dim": 8,
}

TARGET_COLUMNS = [
    "target_next_day_return",
    "target_5d_return",
    "target_10d_return",
    "target_30d_return",
]
TARGET_META = {
    "target_next_day_return": ("1d", "1-DAY", 1),
    "target_5d_return": ("5d", "5-DAY", 5),
    "target_10d_return": ("10d", "10-DAY", 10),
    "target_30d_return": ("30d", "30-DAY", 30),
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


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Data ───────────────────────────────────────────────────────────────────────

def load_split(path):
    df = pd.read_csv(path); df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)

def build_ticker_map(df):
    return {t: i for i, t in enumerate(sorted(df["ticker"].unique()))}

def build_sequences(df, scaler, ticker_map, target, lookback, fit=False):
    feats = df[FEATURE_COLUMNS].values.astype(np.float32)
    if fit: scaler.fit(feats)
    fs = scaler.transform(feats).astype(np.float32)
    tgts = df[target].values.astype(np.float32)
    df = df.reset_index(drop=True); df["_i"] = df.index
    X, T, Y = [], [], []
    for tk, g in df.groupby("ticker"):
        idx = g["_i"].values; tid = ticker_map.get(tk, 0)
        tf, tt = fs[idx], tgts[idx]
        for i in range(lookback, len(tf)):
            X.append(tf[i-lookback:i]); T.append(tid); Y.append(tt[i])
    return np.array(X, dtype=np.float32), np.array(T, dtype=np.int64), np.array(Y, dtype=np.float32)


class SeqDS(Dataset):
    def __init__(self, X, T, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.T[i], self.Y[i]


# ── Model ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, n_tickers, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(n_tickers, embedding_dim)
        self.lstm = nn.LSTM(input_size + embedding_dim, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, tid):
        e = self.emb(tid).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.lstm(torch.cat([x, e], dim=-1))
        return self.fc(out[:, -1, :]).squeeze(-1)


# ── Metrics ────────────────────────────────────────────────────────────────────

def dir_acc(yt, yp): return float((np.sign(yt) == np.sign(yp)).mean())

def information_coefficient(df, preds, target):
    """Average daily Spearman rank IC between predictions and actuals."""
    df = df.copy(); df["y_pred"] = preds
    ics = []
    for _, g in df.groupby("Date"):
        if len(g) < 5: continue
        ic, _ = spearmanr(g[target].values, g["y_pred"].values)
        if not np.isnan(ic): ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0

def compute_metrics(yt, yp):
    return {"mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "directional_accuracy": dir_acc(yt, yp)}

def per_ticker_metrics(df, preds, target):
    df = df.copy(); df["y_pred"] = preds
    return {t: compute_metrics(g[target].to_numpy(), g["y_pred"].to_numpy()) for t, g in df.groupby("ticker")}

def save_per_ticker_csv(pt, path):
    pd.DataFrame([{"ticker": t, **m} for t, m in sorted(pt.items())]).to_csv(path, index=False)


# ── Backtest ───────────────────────────────────────────────────────────────────

def run_backtest(df, preds, target, holding_days=None, benchmark_df=None):
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


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, crit, device):
    model.train(); total = 0.0
    for X, T, Y in loader:
        X, T, Y = X.to(device), T.to(device), Y.to(device)
        opt.zero_grad(); loss = crit(model(X, T), Y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item() * len(Y)
    return total / len(loader.dataset)

def evaluate(model, loader, crit, device):
    model.eval(); total, preds = 0.0, []
    with torch.no_grad():
        for X, T, Y in loader:
            X, T, Y = X.to(device), T.to(device), Y.to(device)
            p = model(X, T); total += crit(p, Y).item()*len(Y); preds.append(p.cpu().numpy())
    return total/len(loader.dataset), np.concatenate(preds)

def align_preds(split_df, preds, lookback, target):
    rows = []
    keep_cols = ["Date", "ticker", target]
    for extra in ("adj_close", "target_next_day_return"):
        if extra in split_df.columns and extra not in keep_cols:
            keep_cols.append(extra)
    for _, g in split_df.groupby("ticker"):
        rows.append(g.iloc[lookback:][keep_cols])
    ref = pd.concat(rows).reset_index(drop=True)
    ref["y_true"] = ref[target].values; ref["y_pred"] = preds
    out_cols = ["Date", "ticker", "y_true", "y_pred"]
    out_cols += [c for c in ("adj_close", "target_next_day_return") if c in ref.columns and c != target]
    return ref[out_cols]


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_results(label, train_m, val_m, test_m, bt):
    print(f"\n  {label} — Statistical Metrics:")
    print(f"  {'Split':<12} {'MAE':<12} {'RMSE':<12} {'Dir Acc'}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*10}")
    for n, m in [("Train", train_m), ("Validation", val_m), ("Test", test_m)]:
        print(f"  {n:<12} {m['mae']:<12.6f} {m['rmse']:<12.6f} {m['directional_accuracy']:.4f}")
    if "error" not in bt:
        lo, ls, bh = bt["long_only"], bt["long_short"], bt["buy_and_hold_benchmark"]
        print(f"\n  Backtest (${INITIAL_CAPITAL:,.0f}):")
        print(f"    Long Only:    ${lo['final_value']:>9,.0f} ({lo['total_return']*100:+.1f}%)  Sharpe: {lo['sharpe_ratio']:.3f}  MaxDD: {lo['max_drawdown']*100:+.1f}%")
        print(f"    Long-Short:   ${ls['final_value']:>9,.0f} ({ls['total_return']*100:+.1f}%)  Sharpe: {ls['sharpe_ratio']:.3f}  MaxDD: {ls['max_drawdown']*100:+.1f}%")
        print(f"    Buy-and-Hold: ${bh['final_value']:>9,.0f} ({bh['total_return']*100:+.1f}%)  Sharpe: {bh['sharpe_ratio']:.3f}  MaxDD: {bh['max_drawdown']*100:+.1f}%")


# ── Run one config ─────────────────────────────────────────────────────────────

def run_lstm(config, config_name, out_dir, train_df, val_df, test_df, ticker_map, device):
    """Train LSTM with given config on both targets, save everything to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tickers = len(ticker_map)
    lookback  = config["lookback"]

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  LSTM — {config_name}")
    print(f"  Config: h={config['hidden_size']} L={config['num_layers']} "
          f"lr={config['learning_rate']} d={config['dropout']}")
    print(f"{'='*60}")

    for target in TARGET_COLUMNS:
        sfx, tgt_label, holding_days = TARGET_META[target]

        print(f"\n── {tgt_label} TARGET ──────────────────────────────────")

        scaler = StandardScaler()
        X_tr, T_tr, Y_tr = build_sequences(train_df, scaler, ticker_map, target, lookback, fit=True)
        X_va, T_va, Y_va = build_sequences(val_df,   scaler, ticker_map, target, lookback)
        X_te, T_te, Y_te = build_sequences(test_df,  scaler, ticker_map, target, lookback)

        # Save scaler to shared
        joblib.dump(scaler, SHARED_DIR / "scaler.pkl")

        print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

        tr_loader = DataLoader(SeqDS(X_tr, T_tr, Y_tr), batch_size=config["batch_size"], shuffle=True)
        va_loader = DataLoader(SeqDS(X_va, T_va, Y_va), batch_size=config["batch_size"], shuffle=False)
        te_loader = DataLoader(SeqDS(X_te, T_te, Y_te), batch_size=config["batch_size"], shuffle=False)

        model = LSTMModel(len(FEATURE_COLUMNS), config["hidden_size"], config["num_layers"],
                          config["dropout"], n_tickers, config["embedding_dim"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
        crit = nn.MSELoss()

        best_val, no_imp = float("inf"), 0
        model_path = out_dir / f"model_{sfx}.pt"

        print(f"  Training (max {config['max_epochs']} ep, patience={config['patience']})...")

        for ep in range(1, config["max_epochs"] + 1):
            tl = train_epoch(model, tr_loader, opt, crit, device)
            vl, _ = evaluate(model, va_loader, crit, device)
            sched.step(vl)
            if ep % 5 == 0 or ep == 1:
                print(f"    Epoch {ep:>3}  train={tl:.6f}  val={vl:.6f}")
            if vl < best_val:
                best_val = vl; no_imp = 0; torch.save(model.state_dict(), model_path)
            else:
                no_imp += 1
                if no_imp >= config["patience"]:
                    print(f"    Early stopping at epoch {ep}"); break

        # Evaluate best
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, tr_p = evaluate(model, tr_loader, crit, device)
        _, va_p = evaluate(model, va_loader, crit, device)
        _, te_p = evaluate(model, te_loader, crit, device)

        tr_m = compute_metrics(Y_tr, tr_p)
        va_m = compute_metrics(Y_va, va_p)
        te_m = compute_metrics(Y_te, te_p)

        # IC on test — measures cross-sectional ranking ability
        te_aligned = align_preds(test_df, te_p, lookback, target)
        ic = information_coefficient(te_aligned, te_aligned["y_pred"].values, "y_true")
        te_m["information_coefficient"] = ic

        # Backtest
        bt = run_backtest(te_aligned, te_aligned["y_pred"].values, "y_true",
                          holding_days=holding_days,
                          benchmark_df=test_df)

        print_results(f"LSTM {config_name} {tgt_label}", tr_m, va_m, te_m, bt)

        # Per-ticker
        pt = per_ticker_metrics(te_aligned.rename(columns={"y_true": target}), te_p, target)
        save_per_ticker_csv(pt, out_dir / f"per_ticker_{sfx}.csv")

        # Save predictions
        for sname, sdf, sp in [("train", train_df, tr_p), ("val", val_df, va_p), ("test", test_df, te_p)]:
            al = align_preds(sdf, sp, lookback, target)
            al.to_csv(out_dir / f"predictions_{sfx}_{sname}.csv", index=False)

        # Save metrics JSON
        results = {"model": f"lstm_{config_name}", "target": target, "config": config,
                   "train": tr_m, "validation": va_m, "test": te_m, "test_backtest": bt}
        with open(out_dir / f"metrics_{sfx}.json", "w") as f:
            json.dump(results, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    ticker_map = build_ticker_map(train_df)
    with open(SHARED_DIR / "ticker_map.json", "w") as f:
        json.dump(ticker_map, f, indent=2)

    # ── Run 1: Default config ──────────────────────────────────────────────────
    run_lstm(DEFAULT_CONFIG, "default", OUTPUT_BASE / "default",
             train_df, val_df, test_df, ticker_map, device)

    # ── Run 2: Tuned config (if exists) ────────────────────────────────────────
    if TUNED_CFG.exists():
        with open(TUNED_CFG) as f:
            tuned_config = json.load(f)
        # Ensure max_epochs and patience are reasonable for final training
        tuned_config.setdefault("max_epochs", 50)
        tuned_config.setdefault("patience", 10)
        tuned_config.setdefault("embedding_dim", 8)

        run_lstm(tuned_config, "tuned", OUTPUT_BASE / "tuned",
                 train_df, val_df, test_df, ticker_map, device)
    else:
        print(f"\n  No tuned config found at {TUNED_CFG}")
        print(f"  To use tuned params, place lstm_best_config.json in outputs/shared/")

    print(f"\n  All LSTM results saved to: {OUTPUT_BASE}")
    print(f"  Shared files saved to: {SHARED_DIR}")


if __name__ == "__main__":
    main()
