"""
src/match.py
Match each target ETF to its best proxy candidates.

OUTPUT CSV columns:
target, proxy1, score1, proxy2, score2, proxy3, score3, proxy4, score4, proxy5, score5
"""

import pandas as pd
from pathlib import Path
from nlp_tags  import load_profiles, build_profiles, save_profiles
from shortlist import shortlist
from scorer    import score, PRICES       # PRICES already loaded inside scorer.py

# ---------- parameters ----------
TOP_K = 500                      # how many candidates to keep

# ---------- load universes ----------
target_csv      = "data/targets_354.csv"
targets         = pd.read_csv(target_csv)["ticker"].tolist()

target_profiles = load_profiles("target")
if not target_profiles:                    # cache missing? build now
    target_profiles = build_profiles(targets)
    save_profiles(target_profiles, "target")

proxy_profiles = load_profiles("proxy")    # cached earlier

# ---------- matching ----------
rows = []
for tk in targets:
    meta  = target_profiles.get(tk, {})
    cands = shortlist(meta)
    

    # drop proxies that have no price history
    cands = [p for p in cands if p in PRICES.columns]
   

    if not cands:
        rows.append({"target": tk})        # no match found
        continue

    # score every candidate
    scored = [(p, round(score(tk, p), 3)) for p in cands]
    scored.sort(key=lambda x: x[1], reverse=True)      # high → low
    top_n = scored[:TOP_K]

    # flatten into dict: proxy1, score1, proxy2, score2, ...
    row = {"target": tk}
    for i, (p, s) in enumerate(top_n, start=1):
        row[f"proxy{i}"]  = p
        row[f"score{i}"]  = s
    rows.append(row)

# ---------- save ----------
out_path = Path("data/match_top500.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"✅  wrote results → {out_path}  (rows = {len(rows)})")
