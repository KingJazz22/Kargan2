"""Fetch and pickle-cache the exact daily bars run_basket_swing_breakout.py uses
(same data.fetch_yfinance call, same date range, same symbol universe), so the
original engine and the vectorbt port can be validated against identical data --
removes yfinance non-determinism as a variable in that comparison.

Usage: python fetch_cache_swing_breakout.py
"""
import os
import pickle

from basket import US_SYMBOLS
from data import fetch_yfinance
from test_mtf_oos_symbols import OOS_US_SYMBOLS

START = "2019-01-01"
END = "2026-08-01"
CACHE_DIR = os.path.join("data_cache", "swing_breakout_vbt")


def cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}_{START}_{END}.pkl")


def fetch_and_cache(symbol):
    path = cache_path(symbol)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    df = fetch_yfinance(symbol, START, END)
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


def load_cached(symbol):
    """Read-only accessor for other scripts -- raises if not yet cached."""
    path = cache_path(symbol)
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_symbols = US_SYMBOLS + OOS_US_SYMBOLS
    ok, failed = 0, []
    for symbol in all_symbols:
        try:
            df = fetch_and_cache(symbol)
            print(f"  {symbol:8s} bars={len(df):5d}  {df.index[0].date()} -> {df.index[-1].date()}")
            ok += 1
        except Exception as e:
            print(f"  {symbol:8s} FAILED: {e}")
            failed.append(symbol)

    print(f"\nCached {ok}/{len(all_symbols)} symbols to {CACHE_DIR}/")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
