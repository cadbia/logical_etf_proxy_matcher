"""
build_features.py
Creates features.csv with one row per (target, proxy) pair.
"""
import pandas as pd, itertools, json, duckdb, numpy as np
from pathlib import Path
from rapidfuzz import fuzz
from scorer import _corr, _liquidity, PROFILES   # reuse helpers
from config import PRICE_FILE


# ---------- load universes ----------
gt = pd.read_csv("data/PROXY_MATCHES_TRAINING_DATA(in).csv")[["target", "proxy"]]
targets = [t for t in gt["target"].unique() if isinstance(t, str)]
proxies = [p for p in PROFILES.keys() if isinstance(p, str)]

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

def name_sim(a, b):
    na = PROFILES.get(a, {}).get("name", "")
    nb = PROFILES.get(b, {}).get("name", "")
    return fuzz.token_set_ratio(na, nb) / 100

def tag_mismatch(a, b, tag):
    ta = PROFILES[a].get(tag)
    tb = PROFILES[b].get(tag)
    if ta is None or tb is None:          # ← no penalty when either side missing
        return 0
    return int(ta != tb)

rows = []
for tgt in targets:
    cands = proxies                                   # full cartesian or shortlist(tgt)
    for prx in cands:
        rows.append({
            "target"  : tgt,
            "proxy"   : prx,
            "c30"     : _corr(tgt, prx, 30),
            "c90"     : _corr(tgt, prx, 90),
            "c252"    : _corr(tgt, prx, 252),
            "name"    : name_sim(tgt, prx),
            "liq"     : _liquidity(prx),
            "reg_mis" : tag_mismatch(tgt, prx, "region"),
            "sty_mis" : tag_mismatch(tgt, prx, "style"),
            "thm_mis" : tag_mismatch(tgt, prx, "theme"),
            "mod_mis" : tag_mismatch(tgt, prx, "modifier"),
            "bond_mis": tag_mismatch(tgt, prx, "bond_type"),
        })

df = pd.DataFrame(rows)

# Label = 1 if this (tgt,prx) is the ground-truth pair
gt_set = {(r.target, r.proxy) for r in gt.itertuples()}
df["y"] = df.apply(lambda r: int((r.target, r.proxy) in gt_set), axis=1)

df.to_csv("data/features.csv", index=False)
print("✅  wrote data/features.csv", df.shape)
