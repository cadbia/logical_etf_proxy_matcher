"""
src/shortlist.py  –  super-relaxed candidate generator
No filtering at all—rank *every* proxy by score.
"""
import json
from pathlib import Path

PROXY_PROFILES = json.loads(Path("data/profiles/proxy.json").read_text())

def shortlist(target_meta: dict) -> list[str]:
    """
    Return the full list of proxy tickers, so nothing ever gets filtered
    before the scorer ranks them.
    """
    return list(PROXY_PROFILES.keys())
