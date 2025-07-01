"""
src/shortlist.py
Rule-based candidate filter for ETF proxy matching.
"""

import json
from pathlib import Path

# ---------- load cached proxy profiles once ----------
PROXY_PROFILES = json.loads(
    Path("data/profiles/proxy.json").read_text()
)  # {ticker: {tags}}

# ---------- tag priority list for fallback ----------
FALLBACK_ORDER = [
    ("modifier",),              # drop ESG / Hedged requirement
    ("theme",),                 # drop niche theme, keep sector
    ("country",),               # collapse single-country → region
]

def _matches(meta, target):
    """True if this proxy meta matches every *filled* tag in target."""
    for k, v in target.items():
        if k == "name":               # ignore long name
            continue
        if v is None:
            continue                  # blank tag in target
        if meta.get(k) != v:
            return False
    return True

def shortlist(target_meta: dict):
    """
    Returns a list of proxy tickers that pass the rule filter,
    applying fallback layers if necessary.
    """
    # -------- exact pass ----------
    cands = [p for p, m in PROXY_PROFILES.items() if _matches(m, target_meta)]
    if cands:
        return cands

    # -------- hierarchical fallbacks ----------
    filtered = target_meta.copy()
    for layer in FALLBACK_ORDER:
        for tag in layer:
            filtered.pop(tag, None)         # drop one constraint
        cands = [p for p, m in PROXY_PROFILES.items() if _matches(m, filtered)]
        if cands:
            return cands

    # -------- nothing found ----------
    return []
