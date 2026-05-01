"""
Train TCN (Temporal Convolutional Network) model with default and tuned configs.

TCN uses causal dilated convolutions instead of recurrence (LSTM/GRU).
Advantages: parallelizable (faster training), fixed receptive field,
no vanishing gradient problem, often competitive with RNNs.

Output structure:
    outputs/tcn/default/
        config.json
        metrics_1d.json, metrics_5d.json
        per_ticker_1d.csv, per_ticker_5d.csv
        predictions_1d_test.csv, predictions_5d_test.csv
        model_1d.pt, model_5d.pt
    outputs/tcn/tuned/     (only if tcn_best_config.json exists)
        same structure

Run:  python src/models/7_train_tcn.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


BASE_DIR    = Path(__file__).resolve().parents[2]
TRAIN_PATH  = BASE_DIR / "data" / "splits" / "train.csv"
VAL_PATH    = BASE_DIR / "data" / "splits" / "val.csv"
TEST_PATH   = BASE_DIR / "data" / "splits" / "test.csv"
OUTPUT_BASE = BASE_DIR / "outputs" / "tcn"
SHARED_DIR  = BASE_DIR / "outputs" / "shared"
TUNED_CFG   = SHARED_DIR / "tcn_best_config.json"

RANDOM_SEED = 42
INITIAL_CAPITAL       = 10_000.0
TOP_K_LONGS           = 5
BOTTOM_K_SHORTS       = 5
TRADING_DAYS_PER_YEAR = 252

DEFAULT_CONFIG = {
    "lookback":       20,
    "num_channels":   [64, 64, 64],   # 3 layers of 64 filters each
    "kernel_size":    3,               # convolution window size
    "dropout":        0.2,
    "batch_size":     64,
    "learning_rate":  5e-4,
    "max_epochs":     50,
    "patience":       10,
    "embedding_dim":  8,
}

TARGET_COLUMNS = ["target_next_day_return", "target_5d_return"]
FEATURE_COLUMNS = [
    "log_return", "return_5d", "return_10d",
    "volume_change", "volume_ma_ratio", "obv_change",
    "rsi_14", "macd", "macd_signal", "macd_diff", "rolling_sharpe_20",
    "volatility_10", "atr_14", "bollinger_band_width",
    "spy_log_return", "vix_close", "vix_log_return", "relative_strength",
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


# ── TCN Building Blocks ───────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Causal convolution — pads only on the left so the model cannot
    see future timesteps. This is the core building block of TCN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        # Remove the right-side padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    One residual block of the TCN: two causal convolutions with
    dropout, ReLU, and a residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)

        # 1x1 convolution for residual if channel sizes differ
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.drop(self.relu(self.conv1(x)))
        out = self.drop(self.relu(self.conv2(out)))
        return self.relu(out + self.residual(x))


class TCNBackbone(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.
    Dilation grows as 1, 2, 4, 8, ... giving an exponentially large
    receptive field with few layers.
    """
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ── Full TCN Model ─────────────────────────────────────────────────────────────

class TCNModel(nn.Module):
    """
    TCN with ticker embedding for return prediction.

    Architecture:
        ticker_id → Embedding(n_tickers, embedding_dim)
        embedding concatenated with features at each timestep
        → TCN backbone (causal dilated convolutions)
        → take output at last timestep
        → Linear → scalar prediction
    """
    def __init__(self, input_size, num_channels, kernel_size, dropout,
                 n_tickers, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(n_tickers, embedding_dim)
        self.tcn = TCNBackbone(input_size + embedding_dim, num_channels, kernel_size, dropout)
        self.fc  = nn.Linear(num_channels[-1], 1)

    def forward(self, x, tid):
        # x: (batch, lookback, features) → TCN expects (batch, features, lookback)
        e = self.emb(tid).unsqueeze(1).expand(-1, x.size(1), -1)
        combined = torch.cat([x, e], dim=-1)       # (batch, lookback, features+emb)
        combined = combined.transpose(1, 2)         # (batch, features+emb, lookback)
        out = self.tcn(combined)                    # (batch, channels, lookback)
        return self.fc(out[:, :, -1]).squeeze(-1)   # take last timestep → (batch,)


# ── Metrics ────────────────────────────────────────────────────────────────────

def dir_acc(yt, yp): return float((np.sign(yt) == np.sign(yp)).mean())

def compute_metrics(yt, yp):
    return {"mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "directional_accuracy": dir_acc(yt, yp)}

def information_coefficient(df, preds, target):
    df = df.copy(); df["y_pred"] = preds
    ics = []
    for _, g in df.groupby("Date"):
        if len(g) < 5: continue
        ic, _ = spearmanr(g[target].values, g["y_pred"].values)
        if not np.isnan(ic): ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0

def per_ticker_metrics(df, preds, target):
    df = df.copy(); df["y_pred"] = preds
    return {t: compute_metrics(g[target].to_numpy(), g["y_pred"].to_numpy())
            for t, g in df.groupby("ticker")}

def save_per_ticker_csv(pt, path):
    pd.DataFrame([{"ticker": t, **m} for t, m in sorted(pt.items())]).to_csv(path, index=False)


# ── Backtest ───────────────────────────────────────────────────────────────────

def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(r) > 0 and r.std() > 0 else 0.0

def max_dd(eq):
    pk = np.maximum.accumulate(eq); return float(((eq - pk) / pk).min()) if len(eq) > 0 else 0.0

def run_backtest(df, preds, target):
    df = df.copy(); df["y_pred"] = preds; df["y_true"] = df[target]
    lo, ls, bh = [], [], []
    for _, g in df.groupby("Date"):
        if len(g) < TOP_K_LONGS + BOTTOM_K_SHORTS: continue
        s = g.sort_values("y_pred", ascending=False)
        t = s.head(TOP_K_LONGS)["y_true"].mean()
        b = s.tail(BOTTOM_K_SHORTS)["y_true"].mean()
        lo.append(t); ls.append((t-b)/2.0); bh.append(g["y_true"].mean())
    if not lo: return {"error": "No data"}
    def _b(r):
        si = np.exp(np.array(r))-1; eq = INITIAL_CAPITAL*np.cumprod(1+si)
        return {"final_value": float(eq[-1]), "total_return": float(eq[-1]/INITIAL_CAPITAL-1),
                "sharpe_ratio": sharpe(si), "max_drawdown": max_dd(eq), "n_trading_days": len(si)}
    return {"long_only": _b(lo), "long_short": _b(ls), "buy_and_hold_benchmark": _b(bh)}


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
    for _, g in split_df.groupby("ticker"):
        rows.append(g.iloc[lookback:][["Date", "ticker", target]])
    ref = pd.concat(rows).reset_index(drop=True)
    ref["y_true"] = ref[target].values; ref["y_pred"] = preds
    return ref[["Date", "ticker", "y_true", "y_pred"]]


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_results(label, train_m, val_m, test_m, bt):
    print(f"\n  {label} — Metrics:")
    print(f"  {'Split':<12} {'MAE':<12} {'RMSE':<12} {'Dir Acc':<10} {'IC':<10}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    for n, m in [("Train", train_m), ("Validation", val_m), ("Test", test_m)]:
        ic = m.get("information_coefficient", "")
        ic_str = f"{ic:<10.4f}" if isinstance(ic, float) else f"{'—':<10}"
        print(f"  {n:<12} {m['mae']:<12.6f} {m['rmse']:<12.6f} {m['directional_accuracy']:<10.4f} {ic_str}")
    if "error" not in bt:
        lo, ls, bh = bt["long_only"], bt["long_short"], bt["buy_and_hold_benchmark"]
        print(f"\n  Backtest (${INITIAL_CAPITAL:,.0f}):")
        print(f"    Long Only:    ${lo['final_value']:>9,.0f} ({lo['total_return']*100:+.1f}%)  Sharpe: {lo['sharpe_ratio']:.3f}  MaxDD: {lo['max_drawdown']*100:+.1f}%")
        print(f"    Long-Short:   ${ls['final_value']:>9,.0f} ({ls['total_return']*100:+.1f}%)  Sharpe: {ls['sharpe_ratio']:.3f}  MaxDD: {ls['max_drawdown']*100:+.1f}%")
        print(f"    Buy-and-Hold: ${bh['final_value']:>9,.0f} ({bh['total_return']*100:+.1f}%)  Sharpe: {bh['sharpe_ratio']:.3f}  MaxDD: {bh['max_drawdown']*100:+.1f}%")


# ── Run one config ─────────────────────────────────────────────────────────────

def run_tcn(config, config_name, out_dir, train_df, val_df, test_df, ticker_map, device):
    """Train TCN with given config on both targets, save everything."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tickers = len(ticker_map)
    lookback  = config["lookback"]

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TCN — {config_name}")
    print(f"  Config: channels={config['num_channels']} kernel={config['kernel_size']} "
          f"lr={config['learning_rate']} d={config['dropout']}")
    print(f"{'='*60}")

    for target in TARGET_COLUMNS:
        sfx = "1d" if "next_day" in target else "5d"
        tgt_label = "1-DAY" if sfx == "1d" else "5-DAY"

        print(f"\n── {tgt_label} TARGET ──────────────────────────────────")

        scaler = StandardScaler()
        X_tr, T_tr, Y_tr = build_sequences(train_df, scaler, ticker_map, target, lookback, fit=True)
        X_va, T_va, Y_va = build_sequences(val_df,   scaler, ticker_map, target, lookback)
        X_te, T_te, Y_te = build_sequences(test_df,  scaler, ticker_map, target, lookback)

        print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

        tr_loader = DataLoader(SeqDS(X_tr, T_tr, Y_tr), batch_size=config["batch_size"], shuffle=True)
        va_loader = DataLoader(SeqDS(X_va, T_va, Y_va), batch_size=config["batch_size"], shuffle=False)
        te_loader = DataLoader(SeqDS(X_te, T_te, Y_te), batch_size=config["batch_size"], shuffle=False)

        model = TCNModel(
            input_size    = len(FEATURE_COLUMNS),
            num_channels  = config["num_channels"],
            kernel_size   = config["kernel_size"],
            dropout       = config["dropout"],
            n_tickers     = n_tickers,
            embedding_dim = config["embedding_dim"],
        ).to(device)

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

        model.load_state_dict(torch.load(model_path, map_location=device))
        _, tr_p = evaluate(model, tr_loader, crit, device)
        _, va_p = evaluate(model, va_loader, crit, device)
        _, te_p = evaluate(model, te_loader, crit, device)

        tr_m = compute_metrics(Y_tr, tr_p)
        va_m = compute_metrics(Y_va, va_p)
        te_m = compute_metrics(Y_te, te_p)

        # IC on test
        te_aligned = align_preds(test_df, te_p, lookback, target)
        ic = information_coefficient(te_aligned, te_aligned["y_pred"].values, "y_true")
        te_m["information_coefficient"] = ic

        bt = run_backtest(te_aligned, te_aligned["y_pred"].values, "y_true")

        print_results(f"TCN {config_name} {tgt_label}", tr_m, va_m, te_m, bt)

        # Per-ticker
        pt = per_ticker_metrics(te_aligned.rename(columns={"y_true": target}), te_p, target)
        save_per_ticker_csv(pt, out_dir / f"per_ticker_{sfx}.csv")

        # Save predictions
        for sname, sdf, sp in [("train", train_df, tr_p), ("val", val_df, va_p), ("test", test_df, te_p)]:
            al = align_preds(sdf, sp, lookback, target)
            al.to_csv(out_dir / f"predictions_{sfx}_{sname}.csv", index=False)

        # Save metrics
        results = {"model": f"tcn_{config_name}", "target": target, "config": config,
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

    # Default config
    run_tcn(DEFAULT_CONFIG, "default", OUTPUT_BASE / "default",
            train_df, val_df, test_df, ticker_map, device)

    # Tuned config (if exists)
    if TUNED_CFG.exists():
        with open(TUNED_CFG) as f:
            tuned_config = json.load(f)
        tuned_config.setdefault("max_epochs", 50)
        tuned_config.setdefault("patience", 10)
        tuned_config.setdefault("embedding_dim", 8)

        run_tcn(tuned_config, "tuned", OUTPUT_BASE / "tuned",
                train_df, val_df, test_df, ticker_map, device)
    else:
        print(f"\n  No tuned config found at {TUNED_CFG}")

    print(f"\n  All TCN results saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
