"""
src/scorer.py  –  correlation + name-similarity scorer
"""
import duckdb, numpy as np, pandas as pd
from pathlib import Path
from rapidfuzz import fuzz
from config import PRICE_FILE

# price matrix
PRICES = duckdb.sql(f"SELECT * FROM '{PRICE_FILE}'").fetchdf().set_index("Date")

# cached names for quick lookup
import json
PROFILE_PATH = Path("data/profiles/proxy.json")
NAMES = {k: v["name"] for k, v in json.loads(PROFILE_PATH.read_text()).items()}

def _has(t):         return t in PRICES.columns
def _corr(a, b, w):  # fill_method=None silences future warning
    if not (_has(a) and _has(b)): return 0.0
    j = PRICES[[a, b]].pct_change(fill_method=None).dropna().tail(w)
    return j.corr().iloc[0,1] if not j.empty else 0.0

def score(a: str, b: str) -> float:
    """Higher ⇒ better proxy."""
    c90  = _corr(a, b, 90)
    c252 = _corr(a, b, 252)
    c30  = _corr(a, b, 30)
    # add tiny name-similarity bonus (0-1 scale)
    name_sim = fuzz.token_set_ratio(NAMES.get(a,""), NAMES.get(b,"")) / 100
    return 0.55*c90 + 0.25*c252 + 0.15*c30 + 0.05*name_sim
