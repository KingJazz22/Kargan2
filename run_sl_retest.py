"""Retest the SL_MULT=12.0 / TP=BB(3.0std) exit config across:
  - the short side (independently, entries fitted independently -- never
    assumed symmetric to long, per this project's own long-standing rule)
  - 4h and daily timeframes (in addition to the original 1h confirmation)
  - a range of SL multiples for each (side, timeframe) combination, since the
    hourly-long optimum (12x) has no reason to transfer automatically

Requires prep_dual_entries_cache.py to have been run first for each
timeframe (produces .cache_dual_entries_<tf>/*.parquet with both long_entry
and short_entry populated).

Usage: python run_sl_retest.py [1h|4h|daily]
"""
import glob
import os
import sys

import pandas as pd

import backtest_pcse_final as bt
import exit_lab as el  # for pooled_stats / print_leaderboard-style helpers
from run_exit_sweep import print_leaderboard

SL_MULTS = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
BB_NUM_STD = 3.0


def load_dual_cache(timeframe: str) -> dict:
    out_dir = f".cache_dual_entries_{timeframe}"
    dfs = {}
    for path in glob.glob(os.path.join(out_dir, "*.parquet")):
        symbol = os.path.splitext(os.path.basename(path))[0]
        dfs[symbol] = pd.read_parquet(path)
    return dfs


def run_pooled_side(dfs: dict, side: str, sl_mult: float) -> dict:
    all_trades = []
    for symbol, prep in dfs.items():
        rows = prep.iloc[bt.WARMUP_BARS:]
        if len(rows) < 10:
            continue
        isolated = bt.isolate_side(rows, side)
        trades, _ = bt.simulate_rows(isolated, sl_mult=sl_mult, bb_num_std=BB_NUM_STD)
        all_trades.extend([t for t in trades if t.side == side])
    return el.pooled_stats([{"r_multiple": t.r_multiple(), "pnl": t.pnl()} for t in all_trades])


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    dfs = load_dual_cache(timeframe)
    print(f"[{timeframe}] Loaded {len(dfs)} symbols: {sorted(dfs)}")

    for side in ("long", "short"):
        rows = []
        for sl_mult in SL_MULTS:
            stats = run_pooled_side(dfs, side, sl_mult)
            rows.append({"label": f"{side} SL={sl_mult}x TP=BB({BB_NUM_STD}std)", "stats": stats})
        print_leaderboard(f"[{timeframe}] {side.upper()} side: SL sweep (TP=BB{BB_NUM_STD}std fixed)", rows, top_n=10)


if __name__ == "__main__":
    main()
