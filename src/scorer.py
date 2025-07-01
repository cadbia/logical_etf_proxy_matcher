"""
src/scorer.py
Similarity score = corr + name overlap + liquidity.
"""

import duckdb, numpy as np
from pathlib import Path
from rapidfuzz import fuzz
from config import PRICE_FILE

# ------------------------------------------------------------------
# Load price matrix once
PRICES = (
    duckdb.sql(f"SELECT * FROM '{PRICE_FILE}'")
    .fetchdf()
    .set_index("Date")
)

# Load cached profiles (need names + aum)
import json
PROFILES = {}
for side in ("proxy", "target"):
    p = Path(f"data/profiles/{side}.json")
    if p.exists():
        PROFILES |= json.loads(p.read_text())

# ------------------------------------------------------------------
def _has(ticker: str) -> bool:
    return ticker in PRICES.columns

def _corr(a: str, b: str, window: int) -> float:
    if not (_has(a) and _has(b)):
        return 0.0
    j = (
        PRICES[[a, b]]
        .pct_change(fill_method=None)   # no forward-fill
        .dropna()
        .tail(window)
    )
    return j.corr().iloc[0, 1] if not j.empty else 0.0

def _liquidity(ticker: str) -> float:
    """log-scale AUM ⇒ 0-1; missing ⇒ 0."""
    raw = PROFILES.get(ticker, {}).get("aum", 0)
    return np.log10(raw + 1) / 7        # 10 billion → ~1.0

def _name_sim(a: str, b: str) -> float:
    na = PROFILES.get(a, {}).get("name", "")
    nb = PROFILES.get(b, {}).get("name", "")
    return fuzz.token_set_ratio(na, nb) / 100

def score(a: str, b: str) -> float:
    """
    Composite similarity:
      0.50 * corr(90d) + 0.25 * corr(252d) + 0.10 * corr(30d)
    + 0.15 * name token overlap
    + 0.05 * liquidity bonus (proxy only)
    """
    c90, c252, c30 = _corr(a, b, 90), _corr(a, b, 252), _corr(a, b, 30)
    return (
        0.50 * c90
        + 0.25 * c252
        + 0.10 * c30
        + 0.15 * _name_sim(a, b)
        + 0.05 * _liquidity(b)
    )
