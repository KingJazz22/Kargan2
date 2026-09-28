"""Local parquet cache for live_trading_v2's 4H PCSE bar fetch.

Measured bottleneck: Alpaca's free-tier IEX feed paginates slowly for a deep
(800-day), 20-symbol 4H bar request -- ~200s of a ~230s total run cycle,
dwarfing every other step (daily-bar fetches for 47 symbols: ~4s each; the
walk-forward fit itself: seconds). Caches full history to disk once per
symbol, then only re-fetches bars newer than the last cached timestamp on
every subsequent run -- collapses ~200s to a few seconds.

Railway's filesystem is ephemeral across container restarts (not just within
one container's uptime), so the first run after every redeploy still pays
the full cost once -- same accepted tradeoff live_state_v2.json already has.
Correctness is unaffected: the merged, deduped, sorted result fed to
strategy_pcse_final.prepare() is identical to what a full re-fetch would
produce, just assembled from cache + an incremental delta instead of one
giant request.
"""
import os

import pandas as pd

import broker_alpaca_v2 as broker

CACHE_DIR = os.getenv("BAR_CACHE_DIR_V2", os.path.join(os.path.dirname(__file__), ".cache_alpaca_v2_4h"))
MAX_BARS_KEPT = 2500  # comfortably covers PCSE_BARS_NEEDED (1750) with headroom


def fetch_4h_bars_cached(symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached: dict[str, pd.DataFrame] = {}
    missing = []
    for sym in symbols:
        path = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if len(df):
                cached[sym] = df
                continue
        missing.append(sym)

    if missing:
        print(f"  bar_cache_v2: cold fetch for {len(missing)} symbol(s) with no cache yet -- this will be slow")
        fresh = broker.fetch_4h_bars(missing, lookback_days=lookback_days)
        cached.update(fresh)

    if cached:
        earliest_last_ts = min(df.index[-1] for df in cached.values())
        catchup_days = max((pd.Timestamp.now(tz="UTC") - earliest_last_ts).days + 2, 2)
        incremental = broker.fetch_4h_bars(list(cached.keys()), lookback_days=catchup_days)
        for sym, new_df in incremental.items():
            combined = pd.concat([cached[sym], new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            cached[sym] = combined.tail(MAX_BARS_KEPT)

    for sym, df in cached.items():
        df.to_parquet(os.path.join(CACHE_DIR, f"{sym}.parquet"))

    return cached
