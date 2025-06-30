# src/config.py
from datetime import date, timedelta

DATA_DIR   = "data"
PRICE_FILE = f"{DATA_DIR}/prices.parquet"

# Three years ago from the day you run the script
START_DATE = (date.today() - timedelta(days=365*1)).isoformat()

CHUNK_SIZE = 250