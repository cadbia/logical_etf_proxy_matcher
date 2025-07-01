"""
src/shortlist.py  –  rule-based candidate generator
"""
import json, yaml
from pathlib import Path

PROXY_PROFILES = json.loads(Path("data/profiles/proxy.json").read_text())
CNTRY2REGION   = json.loads(Path("data/country_to_region.json").read_text())

# ─── helpers ─────────────────────────────────────────────────────────────
def _matches(meta, target):
    for k, v in target.items():
        if k == "name" or v is None:
            continue
        if meta.get(k) != v:
            return False
    return True

def _promote_country(target):
    """Return a *new* meta dict with country upgraded to region if mapping exists."""
    t = target.copy()
    country = t.get("country")
    if country:
        region = CNTRY2REGION.get(country.lower())
        if region:
            t["region"] = region
            t.pop("country", None)
    return t

# ─── main API ────────────────────────────────────────────────────────────
def shortlist(target_meta: dict):
    # 1) exact tag match
    cands = [p for p, m in PROXY_PROFILES.items() if _matches(m, target_meta)]
    if cands:
        return cands

    # 2) drop ESG / Hedged modifier & niche theme
    relaxed = {k: v for k, v in target_meta.items() if k not in ("modifier", "theme")}
    cands = [p for p, m in PROXY_PROFILES.items() if _matches(m, relaxed)]
    if cands:
        return cands

    # 3) promote single-country → region
    promoted = _promote_country(relaxed)
    cands = [p for p, m in PROXY_PROFILES.items() if _matches(m, promoted)]
    if cands:
        return cands

    # 4) final safety-net: same asset class + broad region
    min_tags = {k: v for k, v in target_meta.items() if k in ("asset_class", "region")}
    return [p for p, m in PROXY_PROFILES.items() if _matches(m, min_tags)]
