"""
src/scorer.py
Uses the logistic-regression weights learned in data/weights.json
to compute a probability-like score for every (target, proxy) pair.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
from rapidfuzz import fuzz
from config import PRICE_FILE

# ── load price matrix once ───────────────────────────────────────────────
PRICES = (
    duckdb.sql(f"SELECT * FROM '{PRICE_FILE}'")
    .fetchdf()
)

# accept 'Date' or unnamed index column
if "Date" in PRICES.columns:
    PRICES = PRICES.set_index("Date")
else:                                   # fallback for __index_level_0__
    idx_col = PRICES.columns[0]
    PRICES = PRICES.set_index(idx_col).rename_axis("Date")


# ── profiles (names, AUM, tags) ───────────────────────────
PROFILES = {}
for side in ("proxy", "target"):
    p = Path(f"data/profiles/{side}.json")
    if p.exists():
        PROFILES |= json.loads(p.read_text())

# ── learned weights ───────────────────────────────────────
WEIGHTS = json.loads(Path("data/weights.json").read_text())
B0 = WEIGHTS.pop("intercept")          # constant term

# ---- helpers -----------------------------------------------------------
def _has(tk: str) -> bool:
    return tk in PRICES.columns

def _corr(a: str, b: str, win: int) -> float:
    if not (_has(a) and _has(b)):
        return 0.0
    j = (
        PRICES[[a, b]]
        .pct_change(fill_method=None)
        .dropna()
        .tail(win)
    )
    return j.corr().iloc[0, 1] if not j.empty else 0.0

def _liq(tk: str) -> float:
    raw = PROFILES.get(tk, {}).get("aum", 0) or 0
    return np.log10(raw + 1) / 7          # 0-1

def _name_sim(a: str, b: str) -> float:
    na = PROFILES.get(a, {}).get("name", "")
    nb = PROFILES.get(b, {}).get("name", "")
    return fuzz.token_set_ratio(na, nb) / 100

def _mismatch(a: str, b: str, tag: str) -> int:
    ta = PROFILES.get(a, {}).get(tag)
    tb = PROFILES.get(b, {}).get(tag)
    return 0 if (ta is None or tb is None or ta == tb) else 1

# ---- public API --------------------------------------------------------
def score(a: str, b: str) -> float:
    """
    Returns the *logit* score (higher ⇒ more likely ground-truth proxy):
        B0 + Σ w_i * feature_i
    """
    feats = {
        "c90"     : _corr(a, b, 90),
        "c252"    : _corr(a, b, 252),
        "c30"     : _corr(a, b, 30),
        "name"    : _name_sim(a, b),
        "liq"     : _liq(b),
        "reg_mis" : _mismatch(a, b, "region"),
        "sty_mis" : _mismatch(a, b, "style"),
        "thm_mis" : _mismatch(a, b, "theme"),
        "mod_mis" : _mismatch(a, b, "modifier"),
        "bond_mis": _mismatch(a, b, "bond_type"),
    }

    return B0 + sum(WEIGHTS[k] * feats[k] for k in WEIGHTS)
