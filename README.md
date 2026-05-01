# Portfolio Management System

A stock return prediction system that compares baseline and deep learning models
(LSTM, GRU, TCN) on 15 large U.S. stocks. Predicts next-day log returns using
technical indicators, volume, and market context features.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run Order

Run scripts in this exact order:

```bash
python src/dataset/1_download_yahoo_data.py   # download raw data from Yahoo Finance
python src/dataset/2_build_features.py        # engineer features
python src/dataset/3_prepare_model_data.py    # split into train / val / test
python src/models/4_train_baselines.py        # train and evaluate baseline models
python src/models/5_train_lstm.py             # train and evaluate LSTM models
python src/models/6_train_gru.py              # train and evaluate GRU models
python src/models/99_gather_report.py         # gather all model outputs into reports
```

---

## Project Structure
src/
dataset/
1_download_yahoo_data.py   — downloads OHLCV data for 15 stocks + SPY + VIX
2_build_features.py        — builds features and next-day return target
3_prepare_model_data.py    — chronological train / val / test split
models/
4_train_baselines.py       — Historical Mean, Naive, Linear Regression, Random Forest
5_train_lstm.py            — LSTM default/tuned training and evaluation
6_train_gru.py             — GRU default/tuned training and evaluation
99_gather_report.py        — aggregate model and stock x model report tables
data/
raw/                         — one CSV per ticker from Yahoo Finance
processed/                   — one feature CSV per stock
final_model_dataset.csv      — all 15 stocks combined (37,230 rows)
splits/
train.csv                  — 2015-02-20 to 2021-12-31  (25,950 rows)
val.csv                    — 2022-01-03 to 2023-12-29  ( 7,515 rows)
test.csv                   — 2024-01-02 to 2024-12-30  ( 3,765 rows)
outputs/
metrics/                     — model evaluation results as JSON
predictions/                 — model predictions as CSV
documents/                     — dataset and feature documentation

---

## Stock Universe

15 large U.S. stocks across sectors + SPY (market proxy) + VIX (volatility index):

`AAPL MSFT JPM GS V WMT HD MCD JNJ PG KO CVX CAT IBM DIS`

Data source: Yahoo Finance via `yfinance` — period: 2015-01-01 to 2025-01-01

---

## Features

| Feature | Description |
|---|---|
| `log_return` | Daily log return of Adj Close |
| `volume_change` | Daily % change in volume |
| `rsi_14` | 14-day Relative Strength Index |
| `macd` | MACD line (12/26 EMA) |
| `macd_signal` | MACD signal line (9-day EMA) |
| `macd_diff` | MACD histogram |
| `volatility_10` | 10-day rolling std of log returns |
| `spy_log_return` | Market direction proxy |
| `vix_close` | Market fear / volatility regime |
| `vix_log_return` | Rate of change in market fear |

**Target:** `target_next_day_return` — next day's log return

---

## Baseline Results (Test Set 2024)

| Model | MAE | RMSE | Dir Acc |
|---|---|---|---|
| Historical Mean | 0.008990 | 0.012904 | 54.61% |
| Naive Last Return | 0.013016 | 0.018228 | 51.61% |
| Linear Regression | 0.009146 | 0.013025 | 49.00% |
| Random Forest | 0.008990 | 0.012897 | 54.63% |

---

## Next Steps

- [ ] LSTM
- [ ] GRU
- [ ] TCN
- [ ] Portfolio backtesting



# Run from scratch

```zsh
rm -rf data/ outputs/

python src/dataset/1_download_yahoo_data.py

python src/dataset/2_build_features.py

python src/dataset/3_prepare_model_data.py

python src/models/4_train_baselines.py

python src/models/5_train_lstm.py

python src/models/6_train_gru.py
python src/models/7_train_tcn.py
python src/models/99_gather_report.py

```
