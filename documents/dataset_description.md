# Raw Data Description



This file describes the raw data collected for the capstone project before feature engineering.


## Dataset Selection Rationale

For the first version of the project, a set of **15 large, well-known U.S. stocks** was selected from major sectors, together with **SPY** and **VIX** as market context variables. These stocks were chosen because they are highly liquid, widely studied, and provide a manageable starting universe for building and testing the full pipeline.

The project starts with 15 stocks to keep the first modeling stage simple and interpretable. After the data pipeline, feature engineering, modeling, and backtesting process work correctly, the stock universe can be expanded, for example to the full **Dow 30** or another broader set.

The selected features are based on both practical modeling needs and financial literature. They include stock-specific signals such as returns, volume, momentum, and volatility, as well as market-wide context through SPY and VIX. The idea is to begin with a small, defensible, and interpretable feature set, then later test whether adding more features improves results in a meaningful way.

## Source

The data was downloaded from **Yahoo Finance** using the `yfinance` Python library.

## Assets Collected

### Stocks
- **AAPL** — Apple
- **MSFT** — Microsoft
- **JPM** — JPMorgan Chase
- **GS** — Goldman Sachs
- **V** — Visa
- **WMT** — Walmart
- **HD** — Home Depot
- **MCD** — McDonald’s
- **JNJ** — Johnson & Johnson
- **PG** — Procter & Gamble
- **KO** — Coca-Cola
- **CVX** — Chevron
- **CAT** — Caterpillar
- **IBM** — IBM
- **DIS** — Disney

### Market Context
- **SPY** — ETF proxy for the S&P 500
- **^VIX** — CBOE Volatility Index

## Time Period

- **Start:** 2015-01-01
- **End:** 2025-01-01
- **Rows per file:** 2516 trading days

## Raw Columns

Each CSV file contains:

- **Date** — trading day
- **Open** — opening price
- **High** — highest price of the day
- **Low** — lowest price of the day
- **Close** — closing price
- **Adj Close** — adjusted closing price
- **Volume** — number of shares traded that day

## Storage

Each asset was saved as a separate CSV file in:

```bash
data/raw/