"""
src/scorer.py
Compute similarity score between target and proxy ETF.
"""

import duckdb, numpy as np, pandas as pd
from pathlib import Path
from config import PRICE_FILE

# ---------- load price matrix once ----------
PRICES = duckdb.sql(f"SELECT * FROM '{PRICE_FILE}'").fetchdf().set_index("Date")

def _has(ticker: str) -> bool:
    return ticker in PRICES.columns

def _corr(a: str, b: str, window: int) -> float:
    if not (_has(a) and _has(b)):
        return 0.0                      # missing data → neutral score
    joined = (
        PRICES[[a, b]]
        .pct_change(fill_method=None)   # <-- suppress deprecation warning
        .dropna()
        .tail(window)
    )
    return joined.corr().iloc[0, 1] if not joined.empty else 0.0

def _tracking_error(a: str, b: str, window: int = 252) -> float:
    if not (_has(a) and _has(b)):
        return 1.0                      # maximum penalty
    joined = (
        PRICES[[a, b]]
        .pct_change(fill_method=None)
        .dropna()
        .tail(window)
    )
    if joined.empty:
        return 1.0
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    return np.sqrt((diff**2).mean())

def score(a: str, b: str) -> float:
    """
    Composite similarity score.  Higher is better.
    Returns a large negative number if *both* tickers are missing.
    """
    if not _has(a) and not _has(b):
        return -9.9                     # can't score at all
    c30  = _corr(a, b, 30)
    c90  = _corr(a, b, 90)
    c252 = _corr(a, b, 252)
    te   = _tracking_error(a, b)
    return 0.35 * c90 + 0.25 * c252 + 0.15 * c30 - 0.15 * te
