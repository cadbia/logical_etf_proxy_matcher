# --- src/fetch_prices.py  ------------------------------------
import yfinance as yf, time, duckdb, pandas as pd
from pathlib import Path
from config import DATA_DIR, PRICE_FILE, START_DATE, CHUNK_SIZE

Path(DATA_DIR).mkdir(exist_ok=True)

def _pick_close(df):
    """
    Accepts whatever yfinance.download() just returned and
    normalises it to a 2-D frame of close prices only.
    """
    if isinstance(df.columns, pd.MultiIndex):          # most common layout
        # If levels = (Field, Ticker) because group_by="column"
        if "Close" in df.index.names:                  # unlikely but guard
            return df.xs("Close", axis=0)
        # Field on level-0, ticker on level-1
        if "Close" in df.columns.get_level_values(0):
            return df.xs("Close", level=0, axis=1)
        # Ticker on level-0, field on level-1  (group_by="ticker")
        return df.xs("Close", level=1, axis=1)
    else:                                              # already 1 column per ticker
        # Could be OHLC stacked by date—take Close if present
        close_cols = [c for c in df.columns if c.lower() == "close"]
        if close_cols:
            return df[close_cols]
        raise ValueError("Couldn't locate Close prices")

def fetch_all(tickers):
    frames, failed = [], []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        raw   = yf.download(
            tickers=" ".join(chunk),
            period="1y",            # <-- one-year rolling window
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=True,
        )

        try:
            df = _pick_close(raw)
        except ValueError:
            # nothing usable came back
            failed.extend(chunk)
            continue

        # drop any ticker column that is *entirely* NaN
        df = df.dropna(axis=1, how="all")
        missing = set(chunk) - set(df.columns)
        failed.extend(list(missing))

        frames.append(df)
        time.sleep(5)

    if failed:
        print(f"⚠️  skipped {len(failed)} tickers with no data: {failed[:10]} ...")

    return pd.concat(frames, axis=1)

# ----------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    proxy  = pd.read_csv("data/proxy_722.csv")["ticker"].tolist()
    target = pd.read_csv("data/targets_354.csv")["ticker"].tolist()
    prices = fetch_all(list(set(proxy + target)))
    prices.to_parquet(PRICE_FILE, engine="pyarrow")
    print(f"✅  Saved price matrix → {PRICE_FILE}  (shape={prices.shape})")
