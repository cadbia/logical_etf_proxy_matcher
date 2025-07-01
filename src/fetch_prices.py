# ------------------------------------------------------------
# src/fetch_prices.py
# ------------------------------------------------------------
import time
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

from config import DATA_DIR, PRICE_FILE, CHUNK_SIZE

Path(DATA_DIR).mkdir(exist_ok=True)

# ── NYSE business-day calendar for a clean index ─────────────
BUSDAYS = pd.date_range(
    end=pd.Timestamp.today().normalize(),
    periods=252,                # 1-yr window
    freq="B"                    # NYSE business days
)

# ── helpers ──────────────────────────────────────────────────
def _pick_close(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a 2-D (Date × Ticker) frame of Close prices."""
    if isinstance(raw.columns, pd.MultiIndex):
        # group_by="ticker": level-0 = ticker, level-1 = field
        return raw.xs("Close", level=1, axis=1)
    if "Close" in raw.columns:
        return raw[["Close"]].rename(columns={"Close": raw.columns[0]})
    raise ValueError("No Close column found")


def fetch_chunk(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers=" ".join(tickers),
        period="1y",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=True,
    )
    return _pick_close(raw)


def fetch_all(tickers: list[str]) -> pd.DataFrame:
    frames, failed = [], []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        try:
            df = fetch_chunk(chunk)
        except ValueError:
            failed.extend(chunk)
            continue

        # align to NYSE calendar; drop cols that are entirely NaN
        df = (
            df.reindex(BUSDAYS)         # one common index
              .dropna(axis=1, how="all")
        )
        missing = set(chunk) - set(df.columns)
        failed.extend(list(missing))
        frames.append(df)
        time.sleep(4)

    if failed:
        print(f"⚠️ skipped {len(failed)} tickers (no data).")

    # outer-concatenate so every ticker is a separate column
    return pd.concat(frames, axis=1)


def backfill_missing_cols(prices: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """
    For any ticker not yet in `prices`, pull individually (max period),
    reindex to BUSDAYS, and splice in.
    """
    missing = set(targets) - set(prices.columns)
    if not missing:
        return prices

    extras = {}
    for tk in missing:
        try:
            s = (
                yf.Ticker(tk)
                .history(period="max", interval="1d", auto_adjust=True)["Close"]
                .reindex(BUSDAYS)
            )
            if s.notna().sum() > 0:
                extras[tk] = s
        except Exception:
            pass

    if extras:
        extras_df = pd.DataFrame(extras)
        prices = pd.concat([prices, extras_df], axis=1)

    return prices


# ── CLI driver ───────────────────────────────────────────────
if __name__ == "__main__":
    proxy  = pd.read_csv("data/proxy_722.csv")["ticker"].tolist()
    target = pd.read_csv("data/targets_354.csv")["ticker"].tolist()
    universe = list(set(proxy + target))

    prices = fetch_all(universe)
    prices = backfill_missing_cols(prices, universe)

    # 1) optional small gap fill (single-day holes only)
    prices = prices.fillna(method="ffill", limit=1)

    # 2) drop illiquid tickers (≦ 200 obs in 252-day window)
    good_cols = [c for c in prices if prices[c].notna().sum() >= 200]
    prices = prices[good_cols]

    prices.to_parquet(PRICE_FILE, engine="pyarrow")
    print(f"✅ Saved {prices.shape[1]} tickers × {prices.shape[0]} rows → {PRICE_FILE}")
