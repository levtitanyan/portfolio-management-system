"""
Train an LSTM model for next-day stock return prediction.

Uses a pooled dataset of 15 stocks with a 20-day lookback window.
Includes ticker embedding so the model knows which stock it is processing.

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

# ── Configuration ──────────────────────────────────────────────────────────────

LOOKBACK        = 20
HIDDEN_SIZE     = 128       # increased from 64
NUM_LAYERS      = 2
DROPOUT         = 0.2
BATCH_SIZE      = 64
LEARNING_RATE   = 5e-4      # reduced from 1e-3
MAX_EPOCHS      = 50
PATIENCE        = 10        # increased from 7
RANDOM_SEED     = 42
EMBEDDING_DIM   = 8         # ticker embedding size

# ── Columns ────────────────────────────────────────────────────────────────────

TARGET_COLUMN = "target_next_day_return"
FEATURE_COLUMNS = [
    "log_return",
    "volume_change",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "volatility_10",
    "spy_log_return",
    "vix_close",
    "vix_log_return",
]


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Select MPS (Apple Silicon), CUDA, or CPU in that order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_split(csv_path: Path) -> pd.DataFrame:
    """Load one split CSV, parse dates, sort by ticker and Date."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


def build_ticker_map(train_df: pd.DataFrame) -> dict[str, int]:
    """
    Assign a unique integer index to each ticker.
    Built from training data only so the mapping is stable across splits.
    """
    tickers = sorted(train_df["ticker"].unique())
    return {ticker: idx for idx, ticker in enumerate(tickers)}


# ── Sequence builder ───────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler,
    ticker_map: dict[str, int],
    lookback: int,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a flat DataFrame into (X, ticker_ids, y) arrays for LSTM input.

    For each ticker, slides a window of `lookback` days across the data.
    Sequences are never built across ticker boundaries.

    Returns:
        X          : float32 array of shape (n_samples, lookback, n_features)
        ticker_ids : int64 array of shape (n_samples,)
        y          : float32 array of shape (n_samples,)
    """
    features = df[FEATURE_COLUMNS].values.astype(np.float32)

    if fit_scaler:
        scaler.fit(features)

    features_scaled = scaler.transform(features).astype(np.float32)
    targets = df[TARGET_COLUMN].values.astype(np.float32)

    df = df.reset_index(drop=True)
    df["_idx"] = df.index

    X_list, ticker_list, y_list = [], [], []

    for ticker, group in df.groupby("ticker"):
        idx            = group["_idx"].values
        ticker_id      = ticker_map.get(ticker, 0)
        ticker_feats   = features_scaled[idx]
        ticker_targets = targets[idx]

        for i in range(lookback, len(ticker_feats)):
            X_list.append(ticker_feats[i - lookback : i])
            ticker_list.append(ticker_id)
            y_list.append(ticker_targets[i])

    X          = np.array(X_list,      dtype=np.float32)
    ticker_ids = np.array(ticker_list, dtype=np.int64)
    y          = np.array(y_list,      dtype=np.float32)

    return X, ticker_ids, y


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class ReturnSequenceDataset(Dataset):
    """
    Wraps (X, ticker_ids, y) arrays as a PyTorch Dataset.
    """

    def __init__(
        self,
        X: np.ndarray,
        ticker_ids: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.X          = torch.tensor(X,          dtype=torch.float32)
        self.ticker_ids = torch.tensor(ticker_ids, dtype=torch.long)
        self.y          = torch.tensor(y,          dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.ticker_ids[idx], self.y[idx]


# ── LSTM Model ─────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Stacked LSTM with ticker embedding for return prediction.

    Architecture:
        ticker_id → Embedding(n_tickers, embedding_dim)
        embedding is repeated across all timesteps and concatenated
        with the feature sequence before being passed to the LSTM.

        LSTM input size = n_features + embedding_dim
        → take last timestep output
        → Linear(hidden_size, 1)
        → scalar return prediction
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_tickers: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.ticker_embedding = nn.Embedding(n_tickers, embedding_dim)

        self.lstm = nn.LSTM(
            input_size  = input_size + embedding_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, ticker_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        x          : (batch, lookback, n_features)
        ticker_ids : (batch,)
        output     : (batch,)
        """
        # Get embedding for each sample: (batch, embedding_dim)
        emb = self.ticker_embedding(ticker_ids)

        # Repeat embedding across all timesteps: (batch, lookback, embedding_dim)
        emb_expanded = emb.unsqueeze(1).expand(-1, x.size(1), -1)

        # Concatenate features and embedding: (batch, lookback, n_features + embedding_dim)
        x_combined = torch.cat([x, emb_expanded], dim=-1)

        lstm_out, _ = self.lstm(x_combined)
        last_step   = lstm_out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


# ── Metrics ────────────────────────────────────────────────────────────────────

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where predicted sign matches true sign."""
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and directional accuracy."""
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one full training epoch and return mean loss."""
    model.train()
    total_loss = 0.0

    for X_batch, ticker_batch, y_batch in loader:
        X_batch      = X_batch.to(device)
        ticker_batch = ticker_batch.to(device)
        y_batch      = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch, ticker_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Run inference on a DataLoader and return (mean_loss, predictions)."""
    model.eval()
    total_loss = 0.0
    all_preds  = []

    with torch.no_grad():
        for X_batch, ticker_batch, y_batch in loader:
            X_batch      = X_batch.to(device)
            ticker_batch = ticker_batch.to(device)
            y_batch      = y_batch.to(device)

            preds = model(X_batch, ticker_batch)
            loss  = criterion(preds, y_batch)

            total_loss += loss.item() * len(y_batch)
            all_preds.append(preds.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(all_preds)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Full LSTM training pipeline:
        1. Load splits and build ticker map
        2. Build scaled sequences with ticker IDs
        3. Train with early stopping on validation loss
        4. Evaluate on train, val, and test
        5. Save model, scaler, metrics, and predictions
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading splits...")
    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    ticker_map = build_ticker_map(train_df)
    n_tickers  = len(ticker_map)
    print(f"  Tickers: {n_tickers}  {list(ticker_map.keys())}")

    # ── Build sequences ────────────────────────────────────────────────────────
    print("Building sequences...")
    scaler = StandardScaler()

    X_train, tid_train, y_train = build_sequences(
        train_df, scaler, ticker_map, LOOKBACK, fit_scaler=True
    )
    X_val,   tid_val,   y_val   = build_sequences(
        val_df, scaler, ticker_map, LOOKBACK
    )
    X_test,  tid_test,  y_test  = build_sequences(
        test_df, scaler, ticker_map, LOOKBACK
    )

    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")

    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to: {scaler_path}")

    ticker_map_path = MODELS_DIR / "ticker_map.json"
    with open(ticker_map_path, "w") as f:
        json.dump(ticker_map, f, indent=2)
    print(f"  Ticker map saved to: {ticker_map_path}")

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        ReturnSequenceDataset(X_train, tid_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        ReturnSequenceDataset(X_val, tid_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        ReturnSequenceDataset(X_test, tid_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = LSTMModel(
        input_size    = len(FEATURE_COLUMNS),
        hidden_size   = HIDDEN_SIZE,
        num_layers    = NUM_LAYERS,
        dropout       = DROPOUT,
        n_tickers     = n_tickers,
        embedding_dim = EMBEDDING_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"\nModel architecture:\n{model}\n")

    # ── Training with early stopping ───────────────────────────────────────────
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_model_path   = MODELS_DIR / "lstm_best.pt"

    print(f"Training for up to {MAX_EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _         = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:>3} / {MAX_EPOCHS}  "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ── Evaluate best model ────────────────────────────────────────────────────
    print(f"\nLoading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, train_preds = evaluate(model, train_loader, criterion, device)
    _, val_preds   = evaluate(model, val_loader,   criterion, device)
    _, test_preds  = evaluate(model, test_loader,  criterion, device)

    train_metrics = compute_metrics(y_train, train_preds)
    val_metrics   = compute_metrics(y_val,   val_preds)
    test_metrics  = compute_metrics(y_test,  test_preds)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n── LSTM Results ──────────────────────────────────────")
    print(f"{'Split':<12} {'MAE':<12} {'RMSE':<12} {'Dir Acc'}")
    print(f"{'─'*12} {'─'*12} {'─'*12} {'─'*10}")
    for name, m in [
        ("Train",      train_metrics),
        ("Validation", val_metrics),
        ("Test",       test_metrics),
    ]:
        print(
            f"{name:<12} {m['mae']:<12.6f} {m['rmse']:<12.6f} "
            f"{m['directional_accuracy']:.4f}"
        )

    # ── Save metrics ───────────────────────────────────────────────────────────
    results = {
        "model": "lstm",
        "config": {
            "lookback":       LOOKBACK,
            "hidden_size":    HIDDEN_SIZE,
            "num_layers":     NUM_LAYERS,
            "dropout":        DROPOUT,
            "batch_size":     BATCH_SIZE,
            "learning_rate":  LEARNING_RATE,
            "max_epochs":     MAX_EPOCHS,
            "patience":       PATIENCE,
            "embedding_dim":  EMBEDDING_DIM,
        },
        "train":      train_metrics,
        "validation": val_metrics,
        "test":       test_metrics,
    }

    metrics_path = METRICS_DIR / "lstm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Save predictions ───────────────────────────────────────────────────────
    for split_name, split_df, preds in [
        ("train", train_df, train_preds),
        ("val",   val_df,   val_preds),
        ("test",  test_df,  test_preds),
    ]:
        ref_rows = []
        for ticker, group in split_df.groupby("ticker"):
            ref_rows.append(
                group.iloc[LOOKBACK:][["Date", "ticker", TARGET_COLUMN]]
            )

        ref_df = pd.concat(ref_rows).reset_index(drop=True)
        ref_df["y_true"] = ref_df[TARGET_COLUMN].values
        ref_df["y_pred"] = preds
        ref_df = ref_df[["Date", "ticker", "y_true", "y_pred"]]

        out_path = PREDICTIONS_DIR / f"lstm_{split_name}_predictions.csv"
        ref_df.to_csv(out_path, index=False)

    print(f"\nSaved metrics     : {metrics_path}")
    print(f"Saved predictions : {PREDICTIONS_DIR}")
    print(f"Saved model       : {best_model_path}")
    print(f"Saved scaler      : {scaler_path}")


if __name__ == "__main__":
    main()