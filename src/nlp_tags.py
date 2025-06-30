"""
src/nlp_tags.py
Creates and caches ETF “profiles” based on keywords in names / descriptions.
"""

from pathlib import Path
import re, json, yaml, spacy, yfinance as yf

# ---------- config ----------
VOCAB_PATH   = Path("data/vocab.yaml")
PROFILE_DIR  = Path("data/profiles")
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load models & vocab ----------
nlp   = spacy.load("en_core_web_sm")          # small English model
VOCAB = yaml.safe_load(VOCAB_PATH.read_text())

# ---------- helper: keyword → tag ----------
def _tag_text(text: str) -> dict:
    text_l = text.lower()
    tags   = {}
    for category, mapping in VOCAB.items():
        for tag, words in mapping.items():
            if any(re.search(rf"\b{w}\b", text_l) for w in words):
                tags[category] = tag
                break                           # stop at first match in this cat
    return tags

# ---------- build profiles ----------
def build_profiles(ticker_list) -> dict:
    """
    Returns {ticker: {"name": str, ...tags}}.
    Makes ONE multi-ticker Yahoo call → fast.
    """
    metas  = {}
    yf_obj = yf.Tickers(" ".join(ticker_list)).tickers
    for tk, sec in yf_obj.items():
        info  = sec.info or {}
        name  = info.get("longName") or info.get("shortName") or tk
        descr = info.get("longBusinessSummary", "")
        metas[tk] = {"name": name, **_tag_text(name + " " + descr)}
    return metas

# ---------- cache helpers ----------
def save_profiles(meta: dict, name: str) -> None:
    out = PROFILE_DIR / f"{name}.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"✅  cached {len(meta):4} profiles → {out}")

def load_profiles(name: str) -> dict:
    p = PROFILE_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}

# ---------- CLI utility ----------
if __name__ == "__main__":
    import argparse, pandas as pd

    ap = argparse.ArgumentParser(description="Build and cache ETF profiles")
    ap.add_argument("--csv",  required=True, help="CSV file with 'ticker' col")
    ap.add_argument("--name", required=True, help="cache name (e.g. proxy)")
    args = ap.parse_args()

    tickers  = pd.read_csv(args.csv)["ticker"].tolist()
    profiles = build_profiles(tickers)
    save_profiles(profiles, args.name)
