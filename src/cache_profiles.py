"""
scripts/cache_profiles.py
One-off script to generate & save profiles for BOTH universes.
"""

import pandas as pd
from nlp_tags import build_profiles, save_profiles

proxy_csv  = "data/proxy_722.csv"
target_csv = "data/targets_354.csv"

proxy_tk   = pd.read_csv(proxy_csv)["ticker"].tolist()
target_tk  = pd.read_csv(target_csv)["ticker"].tolist()

save_profiles(build_profiles(proxy_tk),  "proxy")
save_profiles(build_profiles(target_tk), "target")
