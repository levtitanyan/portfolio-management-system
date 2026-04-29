"""
LSTM Hyperparameter Grid Search

Searches over key hyperparameters and selects the best configuration
based on validation loss. Saves the best config to a JSON file that
5_train_lstm.py automatically loads.

Designed to run on Google Colab with GPU for speed.
On Colab, upload your data/splits/ folder, then run this script.

Search space (24 combinations, ~2 hours on Colab GPU):
    hidden_size   : [64, 128]
    num_layers    : [1, 2]
    learning_rate : [0.001, 0.0005, 0.0001]
    dropout       : [0.2, 0.3]

Fixed parameters:
    lookback      : 20
    batch_size    : 64
    max_epochs    : 30  (reduced for speed — early stopping still active)
    patience      : 7
    embedding_dim : 8

Run from project root:
    python src/models/tune_lstm.py

Or on Colab:
    !python tune_lstm.py
"""

from pathlib import Path
import json
import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ── Paths (adjust if running on Colab) ─────────────────────────────────────────

# Detect if running from project root or from src/models/
if Path("data/splits/train.csv").exists():
    BASE_DIR = Path(".")
elif Path("../../data/splits/train.csv").exists():
    BASE_DIR = Path("../..")
else:
    BASE_DIR = Path(".")

TRAIN_PATH  = BASE_DIR / "data" / "splits" / "train.csv"
VAL_PATH    = BASE_DIR / "data" / "splits" / "val.csv"
MODELS_DIR  = BASE_DIR / "outputs" / "models"

# ── Search space ───────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "hidden_size":    [64, 128],
    "num_layers":     [1, 2],
    "learning_rate":  [1e-3, 5e-4, 1e-4],
    "dropout":        [0.2, 0.3],
}

FIXED_PARAMS = {
    "lookback":      20,
    "batch_size":    64,
    "max_epochs":    30,
    "patience":      7,
    "embedding_dim": 8,
}

RANDOM_SEED = 42

# Only tune on 1-day target — faster and the more important target
TUNE_TARGET = "target_next_day_return"

# ── Columns ────────────────────────────────────────────────────────────────────

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

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data ───────────────────────────────────────────────────────────────────────

def load_split(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


def build_ticker_map(df):
    tickers = sorted(df["ticker"].unique())
    return {t: i for i, t in enumerate(tickers)}


def build_sequences(df, scaler, ticker_map, target, lookback, fit=False):
    feats = df[FEATURE_COLUMNS].values.astype(np.float32)
    if fit:
        scaler.fit(feats)
    feats = scaler.transform(feats).astype(np.float32)
    targets = df[target].values.astype(np.float32)

    df = df.reset_index(drop=True)
    df["_idx"] = df.index

    X, T, Y = [], [], []
    for ticker, grp in df.groupby("ticker"):
        idx = grp["_idx"].values
        tid = ticker_map.get(ticker, 0)
        tf = feats[idx]
        tt = targets[idx]
        for i in range(lookback, len(tf)):
            X.append(tf[i-lookback:i])
            T.append(tid)
            Y.append(tt[i])

    return np.array(X, dtype=np.float32), np.array(T, dtype=np.int64), np.array(Y, dtype=np.float32)


class SeqDataset(Dataset):
    def __init__(self, X, T, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.T[i], self.Y[i]


# ── Model ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 n_tickers, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(n_tickers, embedding_dim)
        self.lstm = nn.LSTM(
            input_size + embedding_dim, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, tid):
        e = self.emb(tid).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.lstm(torch.cat([x, e], dim=-1))
        return self.fc(out[:, -1, :]).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_config(config, train_loader, val_loader, n_tickers, device):
    """Train one LSTM configuration and return best validation loss."""
    torch.manual_seed(RANDOM_SEED)

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

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, config["max_epochs"] + 1):
        # Train
        model.train()
        for X, T, Y in train_loader:
            X, T, Y = X.to(device), T.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X, T), Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, T, Y in val_loader:
                X, T, Y = X.to(device), T.to(device), Y.to(device)
                val_loss += criterion(model(X, T), Y).item() * len(Y)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    return best_val, epoch


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)

    ticker_map = build_ticker_map(train_df)
    n_tickers  = len(ticker_map)

    lookback = FIXED_PARAMS["lookback"]

    scaler = StandardScaler()
    X_train, T_train, Y_train = build_sequences(
        train_df, scaler, ticker_map, TUNE_TARGET, lookback, fit=True
    )
    X_val, T_val, Y_val = build_sequences(
        val_df, scaler, ticker_map, TUNE_TARGET, lookback
    )

    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    batch_size = FIXED_PARAMS["batch_size"]
    train_loader = DataLoader(
        SeqDataset(X_train, T_train, Y_train),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        SeqDataset(X_val, T_val, Y_val),
        batch_size=batch_size, shuffle=False,
    )

    # Generate all combinations
    keys = list(SEARCH_SPACE.keys())
    combos = list(itertools.product(*[SEARCH_SPACE[k] for k in keys]))
    total = len(combos)

    print(f"\n── Grid Search: {total} configurations ──────────────────")
    print(f"Search space: {json.dumps(SEARCH_SPACE, indent=2)}")

    results = []

    for i, values in enumerate(combos, 1):
        config = {k: v for k, v in zip(keys, values)}
        config.update(FIXED_PARAMS)

        start = time.time()
        best_val, stopped_epoch = train_one_config(
            config, train_loader, val_loader, n_tickers, device,
        )
        elapsed = time.time() - start

        results.append({
            "config": config,
            "val_loss": best_val,
            "stopped_epoch": stopped_epoch,
            "time_sec": elapsed,
        })

        print(
            f"  [{i:>2}/{total}]  "
            f"h={config['hidden_size']:<4} "
            f"L={config['num_layers']}  "
            f"lr={config['learning_rate']:<8.5f} "
            f"d={config['dropout']:.1f}  "
            f"val_loss={best_val:.6f}  "
            f"ep={stopped_epoch:<3}  "
            f"{elapsed:.0f}s"
        )

    # Find best
    results.sort(key=lambda r: r["val_loss"])
    best = results[0]

    print(f"\n── Best Configuration ───────────────────────────────")
    print(f"  Val loss: {best['val_loss']:.6f}")
    print(f"  Config:   {json.dumps(best['config'], indent=2)}")

    # Save best config
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = MODELS_DIR / "lstm_best_config.json"
    with open(config_path, "w") as f:
        json.dump(best["config"], f, indent=2)
    print(f"  Saved to: {config_path}")

    # Save full results
    full_path = MODELS_DIR / "lstm_grid_search_results.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results: {full_path}")

    # Print top 5
    print(f"\n── Top 5 Configurations ─────────────────────────────")
    for i, r in enumerate(results[:5], 1):
        c = r["config"]
        print(
            f"  {i}. val_loss={r['val_loss']:.6f}  "
            f"h={c['hidden_size']} L={c['num_layers']} "
            f"lr={c['learning_rate']} d={c['dropout']}"
        )


if __name__ == "__main__":
    main()
