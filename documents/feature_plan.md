# Feature Engineering Plan

## Goal

Use today's stock-level, market-level, sector-level, and calendar information to predict forward stock log returns at four horizons: 1 day, 5 days, 10 days, and 30 days.

---

## Prediction Targets (4)

All targets are cumulative log returns, computed as a rolling sum of daily log returns shifted into the future. A log return target at horizon H is equivalent to `log(price[t+H] / price[t])`.

| Target column | Formula | Horizon |
|---|---|---|
| `target_next_day_return` | `log_return.shift(-1)` | 1 trading day |
| `target_5d_return` | `log_return.rolling(5).sum().shift(-5)` | 5 trading days |
| `target_10d_return` | `log_return.rolling(10).sum().shift(-10)` | 10 trading days |
| `target_30d_return` | `log_return.rolling(30).sum().shift(-30)` | 30 trading days |

Because targets look into the future, the last H rows for each stock are dropped after target creation to prevent look-ahead bias.

---

## Input Features (40 total)

### Price Features (4)

| Feature | Formula | Description |
|---|---|---|
| `log_return` | `ln(adj_close_t / adj_close_{t-1})` | Daily log return — primary price signal |
| `return_5d` | `log_return.rolling(5).sum()` | 5-day cumulative return — weekly momentum |
| `return_10d` | `log_return.rolling(10).sum()` | 10-day cumulative return — two-week momentum |
| `return_30d` | `log_return.rolling(30).sum()` | 30-day cumulative return — monthly momentum |

### Volume Features (3)

| Feature | Formula | Description |
|---|---|---|
| `volume_change` | `(volume_t / volume_{t-1}) - 1`, clipped [1%, 99%] | Day-over-day volume change |
| `volume_ma_ratio` | `volume_t / rolling_mean(volume, 20)` | Volume relative to recent average |
| `obv_change` | Daily pct change in On-Balance Volume, clipped [1%, 99%] | Whether volume flow confirms price direction |

### Momentum Features (5)

| Feature | Formula | Description |
|---|---|---|
| `rsi_14` | Standard RSI over 14 days | Overbought (>70) / oversold (<30) signal |
| `macd` | `EMA(12) - EMA(26)` | Short- vs long-term trend comparison |
| `macd_signal` | `EMA(macd, 9)` | Smoothed MACD — momentum change signal |
| `macd_diff` | `macd - macd_signal` | MACD histogram — turning point strength |
| `rolling_sharpe_20` | `mean(log_return, 20) / std(log_return, 20)` | 20-day risk-adjusted momentum |

### Volatility and Risk Features (8)

| Feature | Formula | Description |
|---|---|---|
| `volatility_10` | `std(log_return, 10)` | Recent 10-day price instability |
| `volatility_20` | `std(log_return, 20)` | 20-day realized volatility |
| `volatility_30` | `std(log_return, 30)` | 30-day realized volatility |
| `atr_14` | 14-day Average True Range | Intraday price range volatility |
| `bollinger_band_width` | `(upper - lower) / middle` (20-day bands) | Volatility expansion / contraction regime |
| `beta_60` | `cov(stock, SPY) / var(SPY)` over 60 days | Systematic market sensitivity |
| `idiosyncratic_vol_20` | `std(log_return - beta * spy_log_return, 20)` | Stock-specific risk unexplained by the market |
| `volatility_regime_20` | `volatility_20 / rolling_mean(volatility_20, 60)` | Volatility relative to its own 60-day average |

### Market Context Features (14)

| Feature | Description |
|---|---|
| `spy_log_return` | SPY daily log return — broad market direction |
| `spy_return_5d` | SPY 5-day cumulative return |
| `spy_return_10d` | SPY 10-day cumulative return |
| `spy_return_30d` | SPY 30-day cumulative return |
| `spy_volatility_20` | SPY 20-day realized volatility |
| `qqq_log_return` | QQQ daily log return — Nasdaq / growth proxy |
| `qqq_return_5d` | QQQ 5-day cumulative return |
| `dia_log_return` | DIA daily log return — Dow Jones proxy |
| `dia_return_5d` | DIA 5-day cumulative return |
| `iwm_log_return` | IWM daily log return — small-cap proxy |
| `iwm_return_5d` | IWM 5-day cumulative return |
| `vix_close` | VIX closing level — market fear index |
| `vix_log_return` | VIX daily log return — rising or falling fear |
| `relative_strength` | `log_return - spy_log_return` — stock vs market outperformance |

### Sector Features (5)

Computed as leave-one-out: each stock's sector signal is the equal-weighted average of all other stocks in the same sector, so the target stock does not contaminate its own sector feature.

| Feature | Description |
|---|---|
| `sector_log_return` | Sector average daily log return (excl. self) |
| `sector_return_5d` | Sector average 5-day cumulative return |
| `sector_return_10d` | Sector average 10-day cumulative return |
| `sector_return_30d` | Sector average 30-day cumulative return |
| `sector_relative_strength` | `log_return - sector_log_return` — stock vs sector |

### Calendar Features (1)

| Feature | Values | Description |
|---|---|---|
| `day_of_week` | 0 (Monday) – 4 (Friday) | Captures weekday seasonality effects |

---

## Feature Count Summary

| Group | Count |
|---|---:|
| Price | 4 |
| Volume | 3 |
| Momentum | 5 |
| Volatility / Risk | 8 |
| Market context | 14 |
| Sector | 5 |
| Calendar | 1 |
| **Total** | **40** |

---

## Design Decisions

- **Adj Close** is used for all return calculations to account for stock splits and dividends.
- **Log returns** are used instead of simple returns: they are additive across time, more stationary, and better behaved for regression.
- **Sector features** use leave-one-out averaging to prevent a stock from learning its own future return through the sector signal.
- **Volume outliers** and **OBV outliers** are clipped at the 1st/99th percentile per stock to prevent extreme events from distorting models.
- **Targets use non-overlapping forward windows**: the 5-day target at row t equals the sum of log returns from t+1 to t+5, which equals `log(price[t+5] / price[t])`. This is verified against actual prices.
- **StandardScaler** is fitted only on the training split and applied identically to validation and test — no look-ahead bias from normalization.
