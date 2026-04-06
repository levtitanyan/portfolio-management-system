# Dataset Columns Explanation

This file documents the dataset columns used for feature engineering and stock return prediction in the capstone project.

## Date

- What it is: The trading day for that row.
- How it is created: Taken directly from the raw CSV file for each stock.
- Why it is useful: It keeps the dataset aligned by time so stock, SPY, and VIX features can be matched correctly.

## ticker

- What it is: The stock symbol for the row, such as `AAPL` or `MSFT`.
- How it is created: Taken from the CSV file name.
- Why it is useful: It tells the model which stock the features belong to and allows multiple stocks to be stored in one dataset.

## adj_close

- What it is: The adjusted closing price of the stock.
- How it is created: Taken from the `Adj Close` column in the raw CSV.
- Why it is useful: It adjusts for stock splits and dividends, so it gives a more accurate price series for return calculations.

## volume

- What it is: The number of shares traded on that day.
- How it is created: Taken from the `Volume` column in the raw CSV.
- Why it is useful: Trading volume can reflect market interest, liquidity, and unusual activity.

## log_return

- What it is: The daily log return of the adjusted closing price.
- How it is created: `ln(adj_close_t / adj_close_t-1)`
- Why it is useful: This is the main daily price movement feature and is often more stable for modeling than raw prices.

## volume_change

- What it is: The day-to-day percentage change in trading volume.
- How it is created: `(volume_t / volume_t-1) - 1`
- Why it is useful: It shows how trading activity changed compared with the previous day, which may signal stronger attention or news.

## rsi_14

- What it is: The 14-day Relative Strength Index.
- How it is created: Calculated from recent price movements over a 14-day window.
- Why it is useful: It is a momentum indicator and may help detect overbought or oversold conditions.

## macd

- What it is: The Moving Average Convergence Divergence value.
- How it is created: `EMA(12) - EMA(26)`
- Why it is useful: It is a trend and momentum indicator that compares short-term and long-term price movement.

## macd_signal

- What it is: The signal line for MACD.
- How it is created: 9-day EMA of `macd`
- Why it is useful: It smooths MACD and helps identify whether momentum is strengthening or weakening.

## macd_diff

- What it is: The difference between MACD and its signal line.
- How it is created: `macd - macd_signal`
- Why it is useful: It shows the momentum gap and can help highlight possible turning points.

## volatility_10

- What it is: The rolling 10-day standard deviation of `log_return`.
- How it is created: Standard deviation of the last 10 days of `log_return`
- Why it is useful: It measures recent instability or risk in the stock's price behavior.

## spy_log_return

- What it is: The daily log return of SPY.
- How it is created: `ln(SPY_t / SPY_t-1)`
- Why it is useful: SPY is a market proxy for the S&P 500, so this feature helps the model capture overall market movement.

## vix_close

- What it is: The VIX level on that day.
- How it is created: Taken from the VIX closing price series.
- Why it is useful: VIX is the market fear and volatility index, so it helps capture the current market regime.

## vix_log_return

- What it is: The daily log return of VIX.
- How it is created: `ln(VIX_t / VIX_t-1)`
- Why it is useful: It captures the daily change in market fear or expected volatility.

## target_next_day_return

- What it is: The stock's log return for the next trading day.
- How it is created: Created by shifting `log_return` by `-1`.
- Why it is useful: This is the target variable the model tries to predict using today's available information.

## How the Model Uses This Dataset

- `X` (inputs) = the feature columns, such as price, volume, technical indicators, and market features.
- `y` (target) = `target_next_day_return`
- Each row means: today's information -> tomorrow's return
