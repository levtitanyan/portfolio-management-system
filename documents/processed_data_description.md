# Processed Data Description

## Purpose

The processed dataset is the modeling-ready output of the feature engineering pipeline. It combines per-stock technical indicators, multi-ETF market context, sector context, risk features, and forward-return targets into a single panel dataset where each row represents one stock on one trading day.

**Row meaning:** today's 40 features → forward return at 1 / 5 / 10 / 30 trading days.

---

## Why the Dataset Starts Later Than Raw Data

Some features require a historical lookback window, so the first valid row for each stock is offset from the raw start date:

| Feature | Lookback needed |
|---|---|
| `log_return` | 1 day |
| `rsi_14` | 14 days |
| `macd` | 26-day EMA warmup |
| `beta_60`, `idiosyncratic_vol_20` | 60 days |
| `rolling_sharpe_20`, `volatility_20` | 20 days |
| `volatility_30`, `return_30d` | 30 days |

Additionally, the last 30 rows per stock are dropped because the 30-day forward target requires 30 future days. In practice the combined dataset runs from approximately 2015-04 to 2025-11 after NaN removal.

---

## Reference Columns

| Column | Description |
|---|---|
| `Date` | Trading date (YYYY-MM-DD) |
| `ticker` | Stock symbol (e.g. `AAPL`) |
| `adj_close` | Dividend- and split-adjusted closing price |
| `volume` | Number of shares traded |

---

## Feature Columns (40)

### Price (4)

| Column | Formula |
|---|---|
| `log_return` | `ln(adj_close_t / adj_close_{t-1})` |
| `return_5d` | `sum(log_return, last 5 days)` |
| `return_10d` | `sum(log_return, last 10 days)` |
| `return_30d` | `sum(log_return, last 30 days)` |

### Volume (3)

| Column | Formula |
|---|---|
| `volume_change` | `(volume_t / volume_{t-1}) - 1`, clipped [1%, 99%] |
| `volume_ma_ratio` | `volume_t / mean(volume, 20 days)` |
| `obv_change` | Daily pct change in On-Balance Volume, clipped [1%, 99%] |

### Momentum (5)

| Column | Formula |
|---|---|
| `rsi_14` | 14-day RSI |
| `macd` | `EMA(adj_close, 12) - EMA(adj_close, 26)` |
| `macd_signal` | `EMA(macd, 9)` |
| `macd_diff` | `macd - macd_signal` |
| `rolling_sharpe_20` | `mean(log_return, 20) / std(log_return, 20)` |

### Volatility and Risk (8)

| Column | Formula |
|---|---|
| `volatility_10` | `std(log_return, 10 days)` |
| `volatility_20` | `std(log_return, 20 days)` |
| `volatility_30` | `std(log_return, 30 days)` |
| `atr_14` | 14-day Average True Range |
| `bollinger_band_width` | `(upper_band - lower_band) / middle_band` (20-day Bollinger) |
| `beta_60` | `cov(log_return, spy_log_return, 60) / var(spy_log_return, 60)` |
| `idiosyncratic_vol_20` | `std(log_return - beta_60 × spy_log_return, 20 days)` |
| `volatility_regime_20` | `volatility_20 / mean(volatility_20, 60 days)` |

### Market Context (14)

| Column | Source |
|---|---|
| `spy_log_return` | SPY daily log return |
| `spy_return_5d` | SPY 5-day cumulative log return |
| `spy_return_10d` | SPY 10-day cumulative log return |
| `spy_return_30d` | SPY 30-day cumulative log return |
| `spy_volatility_20` | SPY 20-day realized volatility |
| `qqq_log_return` | QQQ daily log return |
| `qqq_return_5d` | QQQ 5-day cumulative log return |
| `dia_log_return` | DIA daily log return |
| `dia_return_5d` | DIA 5-day cumulative log return |
| `iwm_log_return` | IWM daily log return |
| `iwm_return_5d` | IWM 5-day cumulative log return |
| `vix_close` | VIX daily closing level |
| `vix_log_return` | VIX daily log return |
| `relative_strength` | `log_return - spy_log_return` |

### Sector (5)

Computed as leave-one-out equal-weighted average of all other stocks in the same sector. This prevents a stock from leaking its own future return into the sector signal.

| Column | Description |
|---|---|
| `sector_log_return` | Average daily log return across sector peers |
| `sector_return_5d` | Average 5-day cumulative return across peers |
| `sector_return_10d` | Average 10-day cumulative return across peers |
| `sector_return_30d` | Average 30-day cumulative return across peers |
| `sector_relative_strength` | `log_return - sector_log_return` |

### Calendar (1)

| Column | Values | Description |
|---|---|---|
| `day_of_week` | 0–4 | 0 = Monday, 4 = Friday — weekday seasonality |

---

## Target Columns (4)

| Column | Formula | Horizon |
|---|---|---|
| `target_next_day_return` | `log_return.shift(-1)` | 1 day |
| `target_5d_return` | `log_return.rolling(5).sum().shift(-5)` | 5 days |
| `target_10d_return` | `log_return.rolling(10).sum().shift(-10)` | 10 days |
| `target_30d_return` | `log_return.rolling(30).sum().shift(-30)` | 30 days |

All targets are log returns. During model evaluation they are converted to simple returns via `expm1()` for portfolio simulation.

---

## Dataset Dimensions

After NaN removal (lookback warmup + forward target rows dropped):

| Universe | Rows | Stocks | Features | Targets |
|---|---:|---:|---:|---:|
| tech30 | ~80,670 | 30 | 40 | 4 |
| energy30 | ~80,670 | 30 | 40 | 4 |

---

## File Locations

```
data/{universe}/
    final_model_dataset.csv    — combined panel (all stocks, all dates)
    processed/
        AAPL.csv               — per-stock processed file (tech30 example)
        ...
    splits/
        train.csv              — 2015 → 2021
        val.csv                — 2022 → 2023
        test.csv               — 2024 → 2026
```
