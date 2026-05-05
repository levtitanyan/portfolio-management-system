# Portfolio Management System

## Overview

After model training, the system simulates realistic trading strategies on the out-of-sample test set (2024-01-02 → 2026-02-17) and produces comprehensive performance reports. The same infrastructure supports both stock universes.

---

## Models

The system trains and evaluates eight models across four prediction horizons:

### Baseline Models

| Model | Type |
|---|---|
| Historical Mean | Predicts the training-set mean return for each stock |
| Naive Last Return | Uses today's return to predict tomorrow (1-day only) |
| Linear Regression | OLS with all 40 features |
| Ridge Regression | L2-regularized linear model |
| Random Forest | Non-linear ensemble with time-weighted training |

### Deep Learning Models

| Model | Architecture |
|---|---|
| LSTM | Long Short-Term Memory with ticker embedding, default + tuned |
| GRU | Gated Recurrent Unit with ticker embedding, default + tuned |
| TCN | Temporal Convolutional Network (causal dilated convolutions), default + tuned |

All deep learning models use:
- A sequence lookback window of previous trading days as input
- A learnable ticker embedding that captures stock-specific behavior
- StandardScaler normalization fitted on the training set only

---

## Prediction Horizons

| Horizon | Target | Rebalance frequency |
|---|---|---|
| 1 day | `target_next_day_return` | Daily |
| 5 day | `target_5d_return` | Every 5 trading days |
| 10 day | `target_10d_return` | Every 10 trading days |
| 30 day | `target_30d_return` | Every 30 trading days (~1 month) |

For multi-day horizons, the portfolio is rebalanced at non-overlapping intervals. Each period's realized return is the actual cumulative log return over that holding period, converted to a simple return.

---

## Evaluation Metrics

### Prediction Quality

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error between predicted and actual log returns |
| RMSE | Root Mean Squared Error |
| Directional Accuracy | Fraction of predictions with the correct sign |
| Information Coefficient (IC) | Average daily Spearman rank correlation between predictions and actual returns across all stocks |

IC is the key trading signal metric. A positive IC means the model's ranking is consistently better than random.

### Portfolio Performance

| Metric | Description |
|---|---|
| Total Return | `(final_value / initial_capital) - 1` |
| Annualized Return | Geometric annualized return |
| Sharpe Ratio | `mean_period_return / std_period_return × √(252 / holding_days)` |
| Sortino Ratio | Sharpe using downside deviation only |
| Max Drawdown | Worst peak-to-trough decline in equity curve |
| Calmar Ratio | Annualized return / |max drawdown| |
| Win Rate | Fraction of rebalancing periods with positive net return |
| Profit Factor | Total gains / total losses |
| Statistical Significance | One-sample t-test on period returns (H₀: mean = 0), reported as p-value |

---

## Portfolio Construction

### Long-Only Strategy

1. Rank all available stocks by predicted return (`y_pred`) on each rebalancing date.
2. Select the top K stocks.
3. Allocate capital equally (`1/K` each).
4. Hold for the horizon length.
5. Realize the actual forward return (`y_true`).
6. Deduct transaction costs.
7. Compound the updated portfolio value into the next period.

### Long-Short Strategy

1. Rank all stocks by predicted return.
2. Long the top K stocks (50% of capital, equally weighted).
3. Short the bottom K stocks (50% of capital, equally weighted).
4. Net exposure is zero (dollar-neutral).
5. Profit when long positions outperform short positions.

### Buy-and-Hold Benchmark

1. Buy the full 30-stock universe at equal dollar weight on the first test date.
2. Hold those exact shares through the end of the test period.
3. This benchmark is horizon-independent (same for all four horizons).

---

## Risk Controls

| Control | Implementation |
|---|---|
| Transaction costs | 0.1% of dollar turnover at each rebalance (`COST_PER_TRADE = 0.001`) |
| Drawdown control | If portfolio falls more than 15% from its peak, position sizes are halved for that period |
| Max position weight | Equal-weight Top-K naturally caps any single position at `1/K` (or `0.5/K` for long-short) |
| Dollar-neutral long-short | Long and short legs each use exactly 50% of portfolio value |

---

## Top-K Grid

`portfolio_backtest.py` tests the following grid for every model and horizon:

| Top-K | Behavior |
|---:|---|
| 3 | Most concentrated — highest signal dependence, highest single-stock risk |
| 5 | Balanced — good default for ranking-quality models |
| 10 | More diversified — lower idiosyncratic risk, weaker per-position signal |
| 15 | Broad — approaches market exposure, hardest to beat buy-and-hold |

---

## Output Structure

All outputs are organized by universe. Running `--universe tech30` and `--universe energy30` produces fully independent output trees.

```
outputs/{universe}/
│
├── baselines/
│   ├── historical_mean/
│   │   ├── metrics_1d.json         — prediction metrics + embedded backtest
│   │   ├── metrics_5d.json
│   │   ├── predictions_1d_test.csv — Date, ticker, y_true, y_pred
│   │   └── per_ticker_1d.csv       — per-stock MAE, RMSE, directional accuracy
│   ├── naive_last_return/
│   ├── linear_regression/
│   ├── ridge_regression/
│   └── random_forest/
│
├── lstm/
│   ├── default/
│   └── tuned/
│
├── gru/
│   ├── default/
│   └── tuned/
│
├── tcn/
│   ├── default/
│   └── tuned/
│
├── shared/
│   ├── scaler.pkl                  — StandardScaler fitted on train split
│   ├── ticker_map.json             — ticker → integer ID for embeddings
│   ├── lstm_best_config.json
│   ├── gru_best_config.json
│   └── tcn_best_config.json
│
└── reports/
    ├── md/
    │   ├── performance_report.md   — model metrics table (all horizons)
    │   └── backtest_report.md      — portfolio simulation results (all horizons)
    │
    ├── model_performance_1d.csv    — aggregate model metrics, 1-day horizon
    ├── model_performance_5d.csv
    ├── model_performance_10d.csv
    ├── model_performance_30d.csv
    ├── stock_model_performance_1d.csv   — per-ticker breakdown
    ├── stock_model_performance_5d.csv
    ├── equity_curves_1d.csv        — portfolio value per rebalancing period
    ├── equity_curves_5d.csv
    ├── equity_curves_10d.csv
    ├── equity_curves_30d.csv
    ├── metrics_1d.csv              — full portfolio grid results
    ├── metrics_5d.csv
    ├── metrics_10d.csv
    ├── metrics_30d.csv
    ├── trades_1d.csv               — per-rebalance trade log
    ├── trades_5d.csv
    ├── trades_10d.csv
    └── trades_30d.csv
```

---

## How to Run

### Full pipeline from scratch

```bash
# Diversified 30-stock universe (default)
python run_all.py --universe tech30

# Pure energy sector universe
python run_all.py --universe energy30
```

### Partial runs

```bash
# Skip data download and feature steps (models only)
python run_all.py --universe tech30 --from baselines

# Skip deep learning (baselines + reports only, much faster)
python run_all.py --universe tech30 --from baselines --skip-dl

# Rebuild reports only (predictions already exist)
python run_all.py --universe tech30 --from refresh
```

### View results

After a run completes, the two main human-readable reports are:

```
outputs/tech30/reports/md/performance_report.md   — model comparison
outputs/tech30/reports/md/backtest_report.md       — portfolio results
```
