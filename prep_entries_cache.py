"""One-time (per symbol) step: freeze PCSE's validated entries + attach exit
indicators, cache to disk so exit_lab's sweeps can run fast (no re-fitting
the walk-forward model per exit variant).

Usage: python prep_entries_cache.py [1h|4h|daily] [subset|full]
"""
import sys
import time

import pandas as pd

import basket
import exit_lab
from run_basket_candle_prob import make_fetch_fn

SUBSET_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XOM", "BTC-USD", "ETH-USD", "DOGE-USD"]


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    scope = sys.argv[2] if len(sys.argv) > 2 else "subset"
    symbols = SUBSET_SYMBOLS if scope == "subset" else (basket.US_SYMBOLS + basket.CRYPTO_SYMBOLS)

    fetch_fn = make_fetch_fn(timeframe)
    out_dir = f".cache_entries_{timeframe}"
    import os
    os.makedirs(out_dir, exist_ok=True)

    for symbol in symbols:
        path = os.path.join(out_dir, f"{symbol.replace('/', '_')}.parquet")
        if os.path.exists(path):
            print(f"  {symbol}: already cached")
            continue
        t0 = time.time()
        try:
            df = fetch_fn(symbol)
            prep = exit_lab.prepare_entries(df)
        except Exception as e:
            print(f"  skip {symbol}: {e}")
            continue
        prep.to_parquet(path)
        print(f"  {symbol}: prepped {len(prep)} bars in {time.time()-t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()
