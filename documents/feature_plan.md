# Feature Plan

This document describes the input features and prediction targets used in the stock return prediction model.

## Goal

Use **today's stock, market, and calendar information** to predict **next-day and 5-day forward stock returns**.

---

## Targets

### target_next_day_return (primary)
- **What:** tomorrow's stock log return
- **Calculation:** `shift(log_return, -1)`
- **Why:** primary prediction target — the model learns today's features to predict tomorrow's move

### target_5d_return (secondary)
- **What:** cumulative 5-day forward log return
- **Calculation:** `shift(rolling_sum(log_return, 5), -5)`
- **Why:** medium-term target — daily noise averages out, directional signal is stronger

---

## Input Features (19 total)

### Price Features (3)

**1. log_return**
- Daily stock return based on adjusted closing price
- `ln(adj_close_t / adj_close_t-1)`
- Main price movement signal; more stationary than raw price

**2. return_5d**
- Rolling 5-day cumulative log return
- `sum(log_return) over past 5 days`
- Captures weekly momentum trend

**3. return_10d**
- Rolling 10-day cumulative log return
- `sum(log_return) over past 10 days`
- Captures longer momentum — some patterns develop over two trading weeks

### Volume Features (3)

**4. volume_change**
- Day-to-day percentage change in trading volume, clipped at 1st/99th percentile
- `(volume_t / volume_t-1) - 1`, clipped to remove extreme outliers
- Shows whether trading activity became stronger or weaker

**5. volume_ma_ratio**
- Today's volume divided by 20-day average volume
- `volume_t / rolling_mean(volume, 20)`
- Detects unusually high or low trading activity relative to recent history

**6. obv_change**
- Daily percentage change in On-Balance Volume, clipped at 1st/99th percentile
- OBV accumulates volume on up days and subtracts on down days
- Tracks whether volume flow is confirming or diverging from price direction

### Momentum Features (5)

**7. rsi_14**
- 14-day Relative Strength Index
- `100 - (100 / (1 + avg_gain / avg_loss))`
- Detects overbought (>70) or oversold (<30) conditions

**8. macd**
- Moving Average Convergence Divergence
- `EMA(12) - EMA(26)`
- Compares short-term trend strength vs long-term trend

**9. macd_signal**
- 9-day EMA of MACD line
- Smoothed version of MACD that helps identify momentum changes

**10. macd_diff**
- MACD minus its signal line (also called MACD histogram)
- `macd - macd_signal`
- Highlights turning points in momentum

**11. rolling_sharpe_20**
- 20-day rolling Sharpe ratio
- `rolling_mean(log_return, 20) / rolling_std(log_return, 20)`
- Risk-adjusted momentum — high value means consistent positive returns with low volatility

### Volatility Features (3)

**12. volatility_10**
- 10-day rolling standard deviation of log returns
- Measures recent price instability and risk

**13. atr_14**
- 14-day Average True Range
- `rolling_mean(max(high-low, |high-prev_close|, |low-prev_close|), 14)`
- Measures daily price range volatility — captures intraday risk that log returns miss

**14. bollinger_band_width**
- Width of 20-day Bollinger Bands relative to the moving average
- `(upper_band - lower_band) / middle_band`
- Captures volatility expansion and contraction regimes

### Market Context Features (4)

**15. spy_log_return**
- Daily log return of SPY (S&P 500 ETF)
- `ln(SPY_t / SPY_t-1)`
- Market-wide direction proxy — tells the model whether the broad market moved up or down

**16. vix_close**
- Daily closing level of the VIX index (CBOE Volatility Index)
- Known as the market's "fear index"
- Higher values signal more expected uncertainty and market stress

**17. vix_log_return**
- Daily log return of VIX
- `ln(VIX_t / VIX_t-1)`
- Shows whether market fear is rising or falling

**18. relative_strength**
- Stock's log return minus SPY log return
- `log_return - spy_log_return`
- Captures whether the stock is outperforming or underperforming the market today

### Calendar Features (1)

**19. day_of_week**
- Day of the trading week (0=Monday, 4=Friday)
- Captures known weekday effects — e.g. the Monday effect (stocks tend to decline on Mondays) and the Friday effect

---

## Data Sources

| Source | Used for |
|---|---|
| Stock Adj Close | log_return, return_5d, return_10d, RSI, MACD, Bollinger, rolling_sharpe, volatility, target |
| Stock OHLC | ATR (requires High, Low, Close) |
| Stock Volume | volume_change, volume_ma_ratio, OBV |
| SPY Adj Close | spy_log_return, relative_strength |
| VIX Adj Close | vix_close, vix_log_return |
| Date | day_of_week |

---

## Feature Categories Summary

| Category | Count | Features |
|---|---|---|
| Price | 3 | log_return, return_5d, return_10d |
| Volume | 3 | volume_change, volume_ma_ratio, obv_change |
| Momentum | 5 | rsi_14, macd, macd_signal, macd_diff, rolling_sharpe_20 |
| Volatility | 3 | volatility_10, atr_14, bollinger_band_width |
| Market context | 4 | spy_log_return, vix_close, vix_log_return, relative_strength |
| Calendar | 1 | day_of_week |
| **Total** | **19** | |

---

## Modeling Logic

Each row represents one stock on one trading day:

**today's 19 features → tomorrow's return (1-day) and next week's return (5-day)**

The model learns from six categories of information:
- How the stock price moved recently (price features)
- How actively the stock was traded (volume features)
- Whether momentum is building or fading (momentum features)
- How volatile the stock has been recently (volatility features)
- What the overall market is doing (market context features)
- What day of the week it is (calendar features)

---

## Design Decisions

- **Adj Close** is used for all return calculations because it adjusts for splits and dividends
- **Volume outliers** are clipped at 1st/99th percentile to prevent extreme values from distorting models
- **OBV outliers** are similarly clipped
- **Log returns** are used instead of simple returns because they are additive and more stable
- **Two targets** allow comparison of 1-day vs 5-day prediction difficulty
- All features are academically grounded and literature-supported
