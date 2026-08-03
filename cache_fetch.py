"""Local disk cache around alpaca_fetch's deep-history fetchers, used only by
PCSE's own dev/research scripts. A single symbol's 10yr hourly history takes
~100s+ to pull from Alpaca; the walk-forward validation loop in this
strategy needs to re-run the same basket many times while iterating on
thresholds, so re-fetching every run would dominate wall-clock time for no
reason (the underlying historical bars don't change within a dev session).
Not used by any other strategy in this repo -- they fetch live by design.
"""
import os

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache_alpaca")


def cached_fetch(fetch_fn, symbol: str, timeframe: str, refresh: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_symbol = symbol.replace("/", "_")
    path = os.path.join(CACHE_DIR, f"{safe_symbol}_{timeframe}.parquet")
    if not refresh and os.path.exists(path):
        return pd.read_parquet(path)
    df = fetch_fn(symbol)
    df.to_parquet(path)
    return df
