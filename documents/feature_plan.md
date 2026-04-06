# Feature Plan

This document describes the planned features for the stock return prediction model.

## Goal

Use **today’s stock and market information** to predict **tomorrow’s stock return**.

## Target

### target_next_day_return
- **What it is:** tomorrow’s stock log return
- **How it is calculated:** shift today’s `log_return` by `-1`
- **Why use it:** this is the value the model tries to predict

---

## Planned Input Features

### 1. log_return
- **What it is:** daily stock return based on adjusted closing price
- **How it is calculated:** `ln(adj_close_t / adj_close_t-1)`
- **Why use it:** main price movement signal; better for modeling than raw price

### 2. volume_change
- **What it is:** day-to-day percentage change in trading volume
- **How it is calculated:** `(volume_t / volume_t-1) - 1`
- **Why use it:** shows whether trading activity became stronger or weaker

### 3. rsi_14
- **What it is:** 14-day Relative Strength Index (RSI), a momentum indicator
- **How it is calculated:** based on average recent gains and losses over 14 days  
  `RS = average gain / average loss`  
  `RSI = 100 - (100 / (1 + RS))`
- **Why use it:** helps detect strong recent buying or selling pressure

### 4. macd
- **What it is:** Moving Average Convergence Divergence (MACD), a trend and momentum indicator
- **How it is calculated:** `EMA(12) - EMA(26)`
- **Why use it:** compares short-term trend with long-term trend

### 5. macd_signal
- **What it is:** signal line of MACD
- **How it is calculated:** `9-day EMA of MACD`
- **Why use it:** smooths MACD and helps track momentum changes

### 6. macd_diff
- **What it is:** difference between MACD and its signal line
- **How it is calculated:** `macd - macd_signal`
- **Why use it:** shows whether momentum is strengthening or weakening

### 7. volatility_10
- **What it is:** recent 10-day volatility of stock returns
- **How it is calculated:** rolling 10-day standard deviation of `log_return`
- **Why use it:** measures recent instability or risk

### 8. spy_log_return
- **What it is:** daily return of SPY
- **What is SPY:** an ETF commonly used as a proxy for the **S&P 500**, so it represents the overall US stock market
- **How it is calculated:** `ln(SPY_t / SPY_t-1)`
- **Why use it:** gives market-wide direction and context

### 9. vix_close
- **What it is:** daily VIX level
- **What is VIX:** the **CBOE Volatility Index**, often called the market’s “fear index”; it reflects expected market volatility
- **How it is calculated:** taken directly from the daily VIX closing series
- **Why use it:** helps capture market stress / uncertainty regime

### 10. vix_log_return
- **What it is:** daily return of VIX
- **What is VIX:** a measure of expected market volatility and fear
- **How it is calculated:** `ln(VIX_t / VIX_t-1)`
- **Why use it:** shows whether market fear is rising or falling

---

## Data Used to Build Features

### Stock data
- **Adj Close** — adjusted closing price
- **Volume** — number of shares traded

### Market context data
- **SPY** — market proxy for the S&P 500
- **VIX** — volatility / fear index

---

## Modeling Logic

Each row means:

**today’s features → tomorrow’s return**

The model learns from:
- stock movement
- trading activity
- momentum and trend
- recent volatility
- overall market direction
- market fear / uncertainty

---

## Notes

- Use **Adj Close** for return calculations
- Keep features small, interpretable, and literature-backed
- Start simple, then improve later if needed