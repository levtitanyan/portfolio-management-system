# Processed Data Description

This file describes the processed dataset created after feature engineering for the stock return prediction and portfolio management capstone project.

## Purpose

The processed dataset is the modeling-ready version of the raw stock data. It combines:

- stock-level features
- market context from **SPY**
- volatility context from **VIX**
- the prediction target for the next trading day

The goal is to use **today's information** to predict **tomorrow's stock return**.

## Output Files

After running the feature engineering pipeline, the project creates:

- **Per-stock processed files** in `data/processed/`
- **One combined modeling dataset** in `data/final_model_dataset.csv`

Each row in the processed dataset represents:

**today's stock and market features -> tomorrow's stock return**

## Why the Processed Dataset Starts Later Than the Raw Data

The processed dataset does not start on the very first raw date because some features need historical lookback windows.

- `log_return` needs the previous day
- `rsi_14` needs recent price history
- `macd`, `macd_signal`, and `macd_diff` need moving averages
- `volatility_10` needs 10 days of returns
- `target_next_day_return` uses a shift of `-1`, so the last row is dropped

Because of this, early rows and the final row are removed after feature creation.

## Processed Columns

### Date

- **What it is:** the trading date for the row
- **How it is created:** taken from the original stock CSV and matched with SPY and VIX by date
- **Why it is useful:** it keeps all stock and market features aligned in time

### ticker

- **What it is:** the stock symbol, such as `AAPL` or `MSFT`
- **How it is created:** taken from the stock file name during the per-stock processing loop
- **Why it is useful:** it identifies which stock each row belongs to and allows all stocks to be combined into one dataset

### adj_close

- **What it is:** the adjusted closing price of the stock
- **How it is created:** taken from the raw `Adj Close` column
- **Why it is useful:** adjusted close accounts for stock splits and dividends, so it is better than raw close for return calculations

### volume

- **What it is:** the number of shares traded that day
- **How it is created:** taken from the raw `Volume` column
- **Why it is useful:** trading activity can provide information about liquidity, attention, and unusual market behavior

### log_return

- **What it is:** the stock's daily log return
- **How it is created:** `ln(adj_close_t / adj_close_t-1)`
- **Why it is useful:** this is the main daily price movement feature and is often easier to model than raw prices

### volume_change

- **What it is:** the day-to-day percentage change in trading volume
- **How it is created:** `(volume_t / volume_t-1) - 1`
- **Why it is useful:** it shows whether trading activity increased or decreased compared with the previous day

### rsi_14

- **What it is:** the 14-day Relative Strength Index
- **How it is created:** calculated from recent gains and losses over a 14-day window
- **Why it is useful:** it is a momentum indicator that may help detect overbought or oversold conditions

### macd

- **What it is:** the Moving Average Convergence Divergence value
- **How it is created:** `EMA(12) - EMA(26)`
- **Why it is useful:** it measures trend and momentum by comparing short-term and long-term moving averages

### macd_signal

- **What it is:** the MACD signal line
- **How it is created:** 9-day EMA of `macd`
- **Why it is useful:** it smooths MACD and helps identify momentum shifts

### macd_diff

- **What it is:** the gap between MACD and its signal line
- **How it is created:** `macd - macd_signal`
- **Why it is useful:** it highlights the strength and direction of momentum changes

### volatility_10

- **What it is:** the rolling 10-day standard deviation of stock log returns
- **How it is created:** standard deviation of the last 10 values of `log_return`
- **Why it is useful:** it measures recent instability or risk in the stock's price movement

### spy_log_return

- **What it is:** the daily log return of SPY
- **How it is created:** `ln(SPY_t / SPY_t-1)`
- **Why it is useful:** SPY is used as a proxy for the overall U.S. stock market, so this feature gives market-wide context

### vix_close

- **What it is:** the VIX level on that trading day
- **How it is created:** taken from the VIX closing series
- **Why it is useful:** VIX is the market fear and volatility index, so it helps capture broader market stress

### vix_log_return

- **What it is:** the daily log return of VIX
- **How it is created:** `ln(VIX_t / VIX_t-1)`
- **Why it is useful:** it shows whether market fear or expected volatility is rising or falling

### target_next_day_return

- **What it is:** the stock's log return for the next trading day
- **How it is created:** created by shifting `log_return` by `-1`
- **Why it is useful:** this is the target variable the model tries to predict

## Why These Features Were Chosen

The processed dataset includes a small and interpretable set of features that capture:

- **price movement** through `log_return`
- **trading activity** through `volume` and `volume_change`
- **momentum and trend** through `rsi_14`, `macd`, `macd_signal`, and `macd_diff`
- **recent risk** through `volatility_10`
- **overall market direction** through `spy_log_return`
- **market fear / uncertainty** through `vix_close` and `vix_log_return`

This gives the model both **stock-specific information** and **market-wide context**.

## How the Model Uses the Processed Dataset

- **X (inputs):** all feature columns except the target
- **y (target):** `target_next_day_return`
- **Row meaning:** today's information is used to predict tomorrow's stock return

This structure makes the dataset suitable for:

- train / validation / test splits
- baseline regression models
- sequence models such as LSTM after sequence preparation
