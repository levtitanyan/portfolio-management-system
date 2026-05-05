"""
Stock universe definitions for the Portfolio Management System.

Two universes are supported. The active one is selected via run_all.py:
    python run_all.py --universe tech30     # 30 diversified blue-chip stocks
    python run_all.py --universe energy30   # 30 pure energy sector stocks

All scripts read PORTFOLIO_UNIVERSE from the environment (set by run_all.py).
"""

from __future__ import annotations

import os
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent

# ── Market reference instruments (same for every universe) ─────────────────────

MARKET_REFERENCES = ["SPY", "QQQ", "DIA", "IWM", "^VIX"]

# ── Universe 1: 30 diversified blue-chip stocks ────────────────────────────────

TECH30 = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "ADBE", "CRM",   # Technology
    "JPM",  "GS",   "V",     "BAC",                    # Financials
    "WMT",  "HD",   "MCD",   "AMZN", "NKE",  "COST",  # Consumer
    "JNJ",  "PFE",  "UNH",   "ABT",                    # Healthcare
    "PG",   "KO",                                       # Staples
    "CVX",  "XOM",                                      # Energy
    "CAT",  "HON",  "BA",                               # Industrials
    "IBM",  "DIS",  "NEE",                              # Other
]

# ── Universe 2: 30 pure energy stocks (all have 2015-2026 data) ───────────────

ENERGY30 = [
    "XOM",  "CVX",  "COP",                              # Integrated / Major
    "EOG",  "OXY",  "DVN",  "HES",  "APA",             # E&P large-cap
    "FANG", "AR",   "MTDR", "SM",   "MUR",  "NOG",     # E&P mid-cap
    "OVV",                                              # E&P (Ovintiv)
    "MPC",  "PSX",  "VLO",  "PBF",                     # Refining
    "KMI",  "WMB",  "OKE",  "ET",   "EPD",             # Midstream pipelines
    "TRGP", "LNG",                                      # Midstream / LNG
    "SLB",  "HAL",  "RIG",                              # Oilfield services
    "NRG",                                              # Power
]

# ── Sector assignments (used by 2_build_features.py) ──────────────────────────

SECTOR_MAP: dict[str, str] = {
    # tech30
    "AAPL": "technology",  "MSFT": "technology",  "GOOGL": "technology",
    "NVDA": "technology",  "ADBE": "technology",  "CRM":   "technology",
    "IBM":  "technology",
    "JPM":  "financials",  "GS":   "financials",  "V":     "financials",
    "BAC":  "financials",
    "WMT":  "consumer",    "HD":   "consumer",    "MCD":   "consumer",
    "AMZN": "consumer",    "NKE":  "consumer",    "COST":  "consumer",
    "JNJ":  "healthcare",  "PFE":  "healthcare",  "UNH":   "healthcare",
    "ABT":  "healthcare",
    "PG":   "staples",     "KO":   "staples",
    "CVX":  "energy",      "XOM":  "energy",
    "CAT":  "industrials", "HON":  "industrials", "BA":    "industrials",
    "DIS":  "communication",
    "NEE":  "utilities",
    # energy30
    "COP":  "energy", "EOG":  "energy", "OXY":  "energy", "DVN":  "energy",
    "HES":  "energy", "APA":  "energy", "FANG": "energy", "AR":   "energy",
    "MTDR": "energy", "SM":   "energy", "MUR":  "energy", "NOG":  "energy",
    "OVV":  "energy", "MPC":  "energy", "PSX":  "energy", "VLO":  "energy",
    "PBF":  "energy", "KMI":  "energy", "WMB":  "energy", "OKE":  "energy",
    "ET":   "energy", "EPD":  "energy", "TRGP": "energy", "LNG":  "energy",
    "SLB":  "energy", "HAL":  "energy", "RIG":  "energy", "NRG":  "energy",
}

# ── Registry ───────────────────────────────────────────────────────────────────

UNIVERSES: dict[str, list[str]] = {
    "tech30":   TECH30,
    "energy30": ENERGY30,
}


def get_universe_name() -> str:
    name = os.environ.get("PORTFOLIO_UNIVERSE", "tech30").strip().lower()
    if name not in UNIVERSES:
        raise ValueError(f"Unknown universe '{name}'. Available: {', '.join(UNIVERSES)}")
    return name


def get_tickers() -> list[str]:
    """Stock tickers for the active universe (no market references)."""
    return list(UNIVERSES[get_universe_name()])


def get_all_download_tickers() -> list[str]:
    """Stock tickers + market references needed for the download step."""
    return get_tickers() + MARKET_REFERENCES


def get_data_dir() -> Path:
    return _BASE_DIR / "data" / get_universe_name()


def get_outputs_dir() -> Path:
    return _BASE_DIR / "outputs" / get_universe_name()
