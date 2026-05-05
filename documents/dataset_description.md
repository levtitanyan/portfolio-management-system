# Dataset Description

## Overview

The project downloads daily OHLCV data for 30 U.S. equity stocks plus five market reference instruments. Two distinct stock universes are supported and can be selected at runtime via `run_all.py --universe <name>`.

## Stock Universes

### Universe 1 — Diversified Blue-Chip (`tech30`)

30 large-cap U.S. stocks across eight sectors:

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, NVDA, ADBE, CRM, IBM |
| Financials | JPM, GS, V, BAC |
| Consumer | WMT, HD, MCD, AMZN, NKE, COST |
| Healthcare | JNJ, PFE, UNH, ABT |
| Staples | PG, KO |
| Energy | CVX, XOM |
| Industrials | CAT, HON, BA |
| Communication / Utilities | DIS, NEE |

### Universe 2 — Pure Energy Sector (`energy30`)

30 U.S. energy stocks across five sub-sectors:

| Sub-sector | Tickers |
|---|---|
| Integrated / Major | XOM, CVX, COP |
| E&P large-cap | EOG, OXY, DVN, HES, APA |
| E&P mid-cap | FANG, AR, MTDR, SM, MUR, NOG, OVV |
| Refining | MPC, PSX, VLO, PBF |
| Midstream / LNG | KMI, WMB, OKE, ET, EPD, TRGP, LNG |
| Oilfield Services | SLB, HAL, RIG |
| Power | NRG |

All 30 energy stocks have uninterrupted Yahoo Finance data from 2015-01-02 onwards. Verified before selection.

## Market Reference Instruments

Downloaded for every universe, used as market context features but not as trading targets:

| Ticker | Description |
|---|---|
| SPY | SPDR S&P 500 ETF — broad market proxy |
| QQQ | Nasdaq 100 ETF — growth / tech sector proxy |
| DIA | Dow Jones Industrial Average ETF |
| IWM | Russell 2000 ETF — small-cap proxy |
| ^VIX | CBOE Volatility Index — market fear gauge |

## Time Period

| Parameter | Value |
|---|---|
| Start | 2015-01-01 |
| End | 2026-04-01 |
| Rows per file | ~2,827 trading days |
| Source | Yahoo Finance via `yfinance` |

## Raw Columns

Each downloaded CSV contains:

| Column | Description |
|---|---|
| Date | Trading date (YYYY-MM-DD) |
| Open | Opening price |
| High | Intraday high |
| Low | Intraday low |
| Close | Closing price |
| Adj Close | Dividend- and split-adjusted closing price |
| Volume | Number of shares traded |

`Adj Close` is used for all return and feature calculations. Raw `Close` is a fallback only when `Adj Close` is unavailable.

## Storage

```
data/{universe}/
    raw/
        AAPL.csv
        MSFT.csv
        ...
        SPY.csv  QQQ.csv  DIA.csv  IWM.csv  VIX.csv
    processed/
        AAPL.csv
        ...
    final_model_dataset.csv
    splits/
        train.csv
        val.csv
        test.csv
```

## Train / Validation / Test Split

Splits are chronological and strictly non-overlapping:

| Split | Date range | Approximate years |
|---|---|---|
| Train | 2015-01-02 → 2021-12-31 | 7 years |
| Validation | 2022-01-01 → 2023-12-31 | 2 years |
| Test | 2024-01-01 → 2026-02-17 | ~2 years |

The scaler (`StandardScaler`) is fitted on the training split only and applied identically to validation and test — no look-ahead bias.
