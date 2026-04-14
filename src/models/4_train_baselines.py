from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# Base project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Input split files
TRAIN_PATH = BASE_DIR / "data" / "splits" / "train.csv"
VAL_PATH   = BASE_DIR / "data" / "splits" / "val.csv"
TEST_PATH  = BASE_DIR / "data" / "splits" / "test.csv"

# Output paths
METRICS_DIR     = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"

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
REFERENCE_COLUMNS = ["Date", "ticker"]


def load_split(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    df = pd.read_csv(csv_path)
    required = REFERENCE_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError(f"{csv_path.name} contains invalid Date values.")
    if df[required].isna().any().any():
        raise ValueError(f"{csv_path.name} contains missing values.")
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


def get_xy(df: pd.DataFrame):
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def compute_per_ticker_metrics(
    df: pd.DataFrame, predictions: np.ndarray
) -> dict:
    """
    Break down metrics by ticker so per-stock performance is visible.
    Essential for the paper's results section.
    """
    df = df.copy()
    df["y_pred"] = predictions
    per_ticker = {}
    for ticker, group in df.groupby("ticker"):
        y_true = group[TARGET_COLUMN].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        per_ticker[ticker] = compute_metrics(y_true, y_pred)
    return per_ticker


def historical_mean_predictor(train_df: pd.DataFrame, target_length: int) -> np.ndarray:
    """
    Predict every future return as the mean return from the training set only.
    This is the simplest possible non-trivial baseline.
    """
    mean_return = train_df[TARGET_COLUMN].mean()
    return np.full(target_length, mean_return, dtype=float)


def last_return_predictor(df: pd.DataFrame) -> np.ndarray:
    """
    Predict tomorrow's return as today's log_return.
    log_return at row t is known at end-of-day t, and target is day t+1.
    This is valid — no look-ahead is involved.
    """
    return df["log_return"].to_numpy(dtype=float)


def build_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit scaler on training data only.
    Must never be fit on val or test to avoid data leakage.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_predictions(
    df: pd.DataFrame, predictions: np.ndarray, model_name: str, split_name: str
) -> None:
    out = df[REFERENCE_COLUMNS].copy()
    out["y_true"] = df[TARGET_COLUMN].values
    out["y_pred"] = predictions
    path = PREDICTIONS_DIR / f"{model_name}_{split_name}_predictions.csv"
    out.to_csv(path, index=False)


def evaluate_model(
    model_name: str,
    train_df: pd.DataFrame, train_preds: np.ndarray,
    val_df:   pd.DataFrame, val_preds:   np.ndarray,
    test_df:  pd.DataFrame, test_preds:  np.ndarray,
) -> dict:
    """
    Evaluate on all three splits.
    Train metrics are included to detect overfitting (especially for Random Forest).
    Per-ticker test metrics are included for the results section of the paper.
    """
    y_train = train_df[TARGET_COLUMN].to_numpy()
    y_val   = val_df[TARGET_COLUMN].to_numpy()
    y_test  = test_df[TARGET_COLUMN].to_numpy()

    return {
        "model":      model_name,
        "train":      compute_metrics(y_train, train_preds),
        "validation": compute_metrics(y_val,   val_preds),
        "test":       compute_metrics(y_test,  test_preds),
        "test_per_ticker": compute_per_ticker_metrics(test_df, test_preds),
    }


def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def print_summary(results: list) -> None:
    for r in results:
        print(f"Model: {r['model']}")
        for split in ("train", "validation", "test"):
            m = r[split]
            print(
                f"  {split.capitalize():<12} -> "
                f"MAE: {m['mae']:.6f}  "
                f"RMSE: {m['rmse']:.6f}  "
                f"Dir Acc: {m['directional_accuracy']:.4f}"
            )
        print()


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    X_train, y_train = get_xy(train_df)
    X_val,   y_val   = get_xy(val_df)
    X_test,  y_test  = get_xy(test_df)

    # Fit scaler on train only — applied to Linear Regression and Random Forest
    scaler = build_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    results = []

    # ── 1. Historical Mean ────────────────────────────────────────────────────
    hist_train_pred = historical_mean_predictor(train_df, len(train_df))
    hist_val_pred   = historical_mean_predictor(train_df, len(val_df))
    hist_test_pred  = historical_mean_predictor(train_df, len(test_df))

    save_predictions(val_df,   hist_val_pred,   "historical_mean", "val")
    save_predictions(test_df,  hist_test_pred,  "historical_mean", "test")

    results.append(evaluate_model(
        "historical_mean",
        train_df, hist_train_pred,
        val_df,   hist_val_pred,
        test_df,  hist_test_pred,
    ))

    # ── 2. Naive Last Return ──────────────────────────────────────────────────
    naive_train_pred = last_return_predictor(train_df)
    naive_val_pred   = last_return_predictor(val_df)
    naive_test_pred  = last_return_predictor(test_df)

    save_predictions(val_df,   naive_val_pred,   "naive_last_return", "val")
    save_predictions(test_df,  naive_test_pred,  "naive_last_return", "test")

    results.append(evaluate_model(
        "naive_last_return",
        train_df, naive_train_pred,
        val_df,   naive_val_pred,
        test_df,  naive_test_pred,
    ))

    # ── 3. Linear Regression ─────────────────────────────────────────────────
    linear_model      = train_linear_regression(X_train_scaled, y_train)
    linear_train_pred = linear_model.predict(X_train_scaled)
    linear_val_pred   = linear_model.predict(X_val_scaled)
    linear_test_pred  = linear_model.predict(X_test_scaled)

    save_predictions(val_df,   linear_val_pred,   "linear_regression", "val")
    save_predictions(test_df,  linear_test_pred,  "linear_regression", "test")

    results.append(evaluate_model(
        "linear_regression",
        train_df, linear_train_pred,
        val_df,   linear_val_pred,
        test_df,  linear_test_pred,
    ))

    # ── 4. Random Forest ─────────────────────────────────────────────────────
    rf_model      = train_random_forest(X_train_scaled, y_train)
    rf_train_pred = rf_model.predict(X_train_scaled)
    rf_val_pred   = rf_model.predict(X_val_scaled)
    rf_test_pred  = rf_model.predict(X_test_scaled)

    save_predictions(val_df,   rf_val_pred,   "random_forest", "val")
    save_predictions(test_df,  rf_test_pred,  "random_forest", "test")

    results.append(evaluate_model(
        "random_forest",
        train_df, rf_train_pred,
        val_df,   rf_val_pred,
        test_df,  rf_test_pred,
    ))

    # ── Save and print ────────────────────────────────────────────────────────
    metrics_path = METRICS_DIR / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Baseline training completed.\n")
    print_summary(results)
    print(f"Saved metrics  : {metrics_path}")
    print(f"Saved predictions: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()