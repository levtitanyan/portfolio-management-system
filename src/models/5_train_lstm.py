"""
Train an LSTM model for stock return prediction.

Supports both 1-day and 5-day targets with 19 features, ticker embeddings,
and integrated backtest evaluation (long-only, long-short, vs buy-and-hold).

Hyperparameters can be overridden via a config JSON file at:
    outputs/models/lstm_best_config.json
If the file exists, those parameters are used (from grid search).
If not, the defaults below are used.

Run from the project root:
    python src/models/5_train_lstm.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parents[2]
TRAIN_PATH      = BASE_DIR / "data" / "splits" / "train.csv"
VAL_PATH        = BASE_DIR / "data" / "splits" / "val.csv"
TEST_PATH       = BASE_DIR / "data" / "splits" / "test.csv"
METRICS_DIR     = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"
MODELS_DIR      = BASE_DIR / "outputs" / "models"
CONFIG_PATH     = MODELS_DIR / "lstm_best_config.json"

# ── Default configuration (overridden by CONFIG_PATH if it exists) ─────────────

DEFAULT_CONFIG = {
    "lookback":       20,
    "hidden_size":    128,
    "num_layers":     2,
    "dropout":        0.2,
    "batch_size":     64,
    "learning_rate":  5e-4,
    "max_epochs":     50,
    "patience":       10,
    "embedding_dim":  8,
}

RANDOM_SEED = 42

# ── Backtest configuration ─────────────────────────────────────────────────────

INITIAL_CAPITAL     = 10_000.0
TOP_K_LONGS         = 5
BOTTOM_K_SHORTS     = 5
TRADING_DAYS_YEAR   = 252

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


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Select CUDA, MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """
    Load hyperparameters from grid search results if available,
    otherwise use defaults.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        print(f"  Loaded tuned config from: {CONFIG_PATH}")
        return config
    print("  Using default config (no tuned config found)")
    return DEFAULT_CONFIG.copy()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_split(csv_path: Path) -> pd.DataFrame:
    """Load one split CSV, parse dates, sort by ticker and Date."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


def build_ticker_map(train_df: pd.DataFrame) -> dict[str, int]:
    """Assign a unique integer index to each ticker from training data."""
    tickers = sorted(train_df["ticker"].unique())
    return {t: i for i, t in enumerate(tickers)}


# ── Sequence builder ───────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler,
    ticker_map: dict[str, int],
    target: str,
    lookback: int,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert flat DataFrame into (X, ticker_ids, y) sequence arrays.
    Sequences are built per-ticker and never cross ticker boundaries.
    """
    features = df[FEATURE_COLUMNS].values.astype(np.float32)

    if fit_scaler:
        scaler.fit(features)

    features_scaled = scaler.transform(features).astype(np.float32)
    targets = df[target].values.astype(np.float32)

    df = df.reset_index(drop=True)
    df["_idx"] = df.index

    X_list, tid_list, y_list = [], [], []

    for ticker, group in df.groupby("ticker"):
        idx          = group["_idx"].values
        ticker_id    = ticker_map.get(ticker, 0)
        t_feats      = features_scaled[idx]
        t_targets    = targets[idx]

        for i in range(lookback, len(t_feats)):
            X_list.append(t_feats[i - lookback : i])
            tid_list.append(ticker_id)
            y_list.append(t_targets[i])

    return (
        np.array(X_list,   dtype=np.float32),
        np.array(tid_list, dtype=np.int64),
        np.array(y_list,   dtype=np.float32),
    )


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class ReturnSequenceDataset(Dataset):
    """Wraps (X, ticker_ids, y) arrays as a PyTorch Dataset."""

    def __init__(self, X, ticker_ids, y):
        self.X   = torch.tensor(X,          dtype=torch.float32)
        self.tid = torch.tensor(ticker_ids, dtype=torch.long)
        self.y   = torch.tensor(y,          dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.tid[idx], self.y[idx]


# ── LSTM Model ─────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Stacked LSTM with ticker embedding.
    Embedding is concatenated with features at each timestep.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 n_tickers, embedding_dim):
        super().__init__()
        self.ticker_embedding = nn.Embedding(n_tickers, embedding_dim)
        self.lstm = nn.LSTM(
            input_size  = input_size + embedding_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, ticker_ids):
        emb = self.ticker_embedding(ticker_ids)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.lstm(torch.cat([x, emb], dim=-1))
        return self.fc(out[:, -1, :]).squeeze(-1)


# ── Metrics ────────────────────────────────────────────────────────────────────

def directional_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def compute_statistical_metrics(y_true, y_pred):
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


# ── Backtest ───────────────────────────────────────────────────────────────────

def compute_sharpe(daily_returns):
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_YEAR))


def compute_max_drawdown(equity):
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def run_backtest(df, predictions, target):
    """
    Run long-only, long-short, and buy-and-hold backtests.
    Returns dict of final values, returns, Sharpe, and max drawdown.
    """
    df = df.copy()
    df["y_pred"] = predictions
    df["y_true"] = df[target]

    lo_rets, ls_rets, bh_rets = [], [], []

    for date, group in df.groupby("Date"):
        if len(group) < TOP_K_LONGS + BOTTOM_K_SHORTS:
            continue
        s = group.sort_values("y_pred", ascending=False)
        top    = s.head(TOP_K_LONGS)["y_true"].mean()
        bottom = s.tail(BOTTOM_K_SHORTS)["y_true"].mean()

        lo_rets.append(top)
        ls_rets.append((top - bottom) / 2.0)
        bh_rets.append(group["y_true"].mean())

    if not lo_rets:
        return {"error": "Not enough data"}

    def _build(rets):
        simple = np.exp(np.array(rets)) - 1
        equity = INITIAL_CAPITAL * np.cumprod(1 + simple)
        return {
            "final_value":   float(equity[-1]),
            "total_return":  float(equity[-1] / INITIAL_CAPITAL - 1),
            "sharpe_ratio":  compute_sharpe(simple),
            "max_drawdown":  compute_max_drawdown(equity),
            "n_trading_days": len(simple),
        }

    return {
        "long_only":              _build(lo_rets),
        "long_short":             _build(ls_rets),
        "buy_and_hold_benchmark": _build(bh_rets),
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for X, tid, y in loader:
        X, tid, y = X.to(device), tid.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X, tid), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total, preds = 0.0, []
    with torch.no_grad():
        for X, tid, y in loader:
            X, tid, y = X.to(device), tid.to(device), y.to(device)
            p = model(X, tid)
            total += criterion(p, y).item() * len(y)
            preds.append(p.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(preds)


# ── Prediction alignment ──────────────────────────────────────────────────────

def align_predictions_to_df(split_df, preds, lookback, target):
    """
    Build a DataFrame with Date, ticker, y_true, y_pred aligned to
    the correct rows (sequences start at row `lookback` per ticker).
    """
    rows = []
    for ticker, group in split_df.groupby("ticker"):
        rows.append(group.iloc[lookback:][["Date", "ticker", target]])
    ref = pd.concat(rows).reset_index(drop=True)
    ref["y_true"] = ref[target].values
    ref["y_pred"] = preds
    return ref[["Date", "ticker", "y_true", "y_pred"]]


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_statistical(name, train_m, val_m, test_m):
    print(f"\n  Statistical Metrics:")
    print(f"  {'Split':<12} {'MAE':<12} {'RMSE':<12} {'Dir Acc'}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*10}")
    for sname, m in [("Train", train_m), ("Validation", val_m), ("Test", test_m)]:
        print(f"  {sname:<12} {m['mae']:<12.6f} {m['rmse']:<12.6f} {m['directional_accuracy']:.4f}")


def print_backtest(bt):
    if "error" in bt:
        print(f"  Backtest error: {bt['error']}")
        return
    lo = bt["long_only"]
    ls = bt["long_short"]
    bh = bt["buy_and_hold_benchmark"]
    print(f"\n  Backtest on Test Set (starting ${INITIAL_CAPITAL:,.0f}):")
    print(f"    Long Only:   ${lo['final_value']:>9,.0f} ({lo['total_return']*100:+.1f}%)  Sharpe: {lo['sharpe_ratio']:.3f}  MaxDD: {lo['max_drawdown']*100:+.1f}%")
    print(f"    Long-Short:  ${ls['final_value']:>9,.0f} ({ls['total_return']*100:+.1f}%)  Sharpe: {ls['sharpe_ratio']:.3f}  MaxDD: {ls['max_drawdown']*100:+.1f}%")
    print(f"    Buy-and-Hold:${bh['final_value']:>9,.0f} ({bh['total_return']*100:+.1f}%)  Sharpe: {bh['sharpe_ratio']:.3f}  MaxDD: {bh['max_drawdown']*100:+.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    config = load_config()

    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # ── Load data ──────────────────────────────────────────────────────────────
    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    ticker_map = build_ticker_map(train_df)
    n_tickers  = len(ticker_map)

    # Save ticker map and scaler
    joblib.dump(None, MODELS_DIR / "scaler.pkl")  # placeholder, replaced below

    ticker_map_path = MODELS_DIR / "ticker_map.json"
    with open(ticker_map_path, "w") as f:
        json.dump(ticker_map, f, indent=2)

    lookback = config["lookback"]
    all_results = []

    for target in TARGET_COLUMNS:
        target_label = "1-DAY" if "next_day" in target else "5-DAY"
        print(f"\n══ LSTM {target_label} TARGET ══════════════════════════════")

        # ── Build sequences ────────────────────────────────────────────────────
        scaler = StandardScaler()
        X_train, tid_train, y_train = build_sequences(
            train_df, scaler, ticker_map, target, lookback, fit_scaler=True
        )
        X_val, tid_val, y_val = build_sequences(
            val_df, scaler, ticker_map, target, lookback
        )
        X_test, tid_test, y_test = build_sequences(
            test_df, scaler, ticker_map, target, lookback
        )

        # Save scaler (last target's scaler is saved — fine since same features)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

        print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

        # ── DataLoaders ────────────────────────────────────────────────────────
        train_loader = DataLoader(
            ReturnSequenceDataset(X_train, tid_train, y_train),
            batch_size=config["batch_size"], shuffle=True,
        )
        val_loader = DataLoader(
            ReturnSequenceDataset(X_val, tid_val, y_val),
            batch_size=config["batch_size"], shuffle=False,
        )
        test_loader = DataLoader(
            ReturnSequenceDataset(X_test, tid_test, y_test),
            batch_size=config["batch_size"], shuffle=False,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        model = LSTMModel(
            input_size    = len(FEATURE_COLUMNS),
            hidden_size   = config["hidden_size"],
            num_layers    = config["num_layers"],
            dropout       = config["dropout"],
            n_tickers     = n_tickers,
            embedding_dim = config["embedding_dim"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
        criterion = nn.MSELoss()

        # ── Training ──────────────────────────────────────────────────────────
        best_val_loss     = float("inf")
        epochs_no_improve = 0
        suffix = "1d" if "next_day" in target else "5d"
        best_path = MODELS_DIR / f"lstm_best_{suffix}.pt"

        print(f"  Training (max {config['max_epochs']} epochs, patience={config['patience']})...")

        for epoch in range(1, config["max_epochs"] + 1):
            t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            v_loss, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(v_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:>3}  train={t_loss:.6f}  val={v_loss:.6f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config["patience"]:
                    print(f"    Early stopping at epoch {epoch}")
                    break

        # ── Evaluate best ──────────────────────────────────────────────────────
        model.load_state_dict(torch.load(best_path, map_location=device))

        _, train_preds = evaluate(model, train_loader, criterion, device)
        _, val_preds   = evaluate(model, val_loader,   criterion, device)
        _, test_preds  = evaluate(model, test_loader,  criterion, device)

        train_m = compute_statistical_metrics(y_train, train_preds)
        val_m   = compute_statistical_metrics(y_val,   val_preds)
        test_m  = compute_statistical_metrics(y_test,  test_preds)

        print_statistical(f"LSTM {target_label}", train_m, val_m, test_m)

        # ── Backtest ──────────────────────────────────────────────────────────
        test_aligned = align_predictions_to_df(test_df, test_preds, lookback, target)
        bt = run_backtest(test_aligned, test_aligned["y_pred"].values, "y_true")
        print_backtest(bt)

        # ── Save predictions ──────────────────────────────────────────────────
        for sname, sdf, spreds in [
            ("train", train_df, train_preds),
            ("val",   val_df,   val_preds),
            ("test",  test_df,  test_preds),
        ]:
            aligned = align_predictions_to_df(sdf, spreds, lookback, target)
            out_path = PREDICTIONS_DIR / f"lstm_{suffix}_{sname}_predictions.csv"
            aligned.to_csv(out_path, index=False)

        # ── Collect results ───────────────────────────────────────────────────
        all_results.append({
            "model":      "lstm",
            "target":     target,
            "config":     config,
            "train":      train_m,
            "validation": val_m,
            "test":       test_m,
            "test_backtest": bt,
        })

    # ── Save all metrics ──────────────────────────────────────────────────────
    metrics_path = METRICS_DIR / "lstm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Saved metrics     : {metrics_path}")
    print(f"  Saved predictions : {PREDICTIONS_DIR}")
    print(f"  Saved models      : {MODELS_DIR}")


if __name__ == "__main__":
    main()
