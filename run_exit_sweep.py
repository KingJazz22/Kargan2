"""Exit-design sweep runner for PCSE. Loads the frozen, cached entry signals
(see prep_entries_cache.py) and tests many exit rules against them, holding
entries fixed throughout. Prints ranked leaderboards for each phase:

  1. Fixed TP x SL (stdev-multiple) grid
  2. TP at first touch of EMA/rolling-VWAP (50/100/150/200), best SL from phase 1
  3. TP at RSI/Stochastic/Bollinger/Keltner levels, best SL from phase 1
  4. The best rule from each phase, run together for a final comparison

Usage: python run_exit_sweep.py [1h|4h|daily]
"""
import glob
import os
import sys

import pandas as pd

import exit_lab as el

SL_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TP_STDEV_MULTS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
RSI_LEVELS = [60, 65, 70, 75, 80]
STOCH_LEVELS = [70, 75, 80, 85, 90]
BB_NUM_STD = [1.5, 2.0, 2.5, 3.0]
KC_MULTS = [1.0, 1.5, 2.0, 2.5]


def load_cache(timeframe: str) -> dict:
    out_dir = f".cache_entries_{timeframe}"
    dfs = {}
    for path in glob.glob(os.path.join(out_dir, "*.parquet")):
        symbol = os.path.splitext(os.path.basename(path))[0]
        dfs[symbol] = pd.read_parquet(path)
    return dfs


def print_leaderboard(title: str, rows: list, top_n: int = 10):
    print(f"\n=== {title} ===")
    ranked = sorted(rows, key=lambda r: r["stats"]["t_stat"], reverse=True)
    print(f"{'label':30s} {'n':>6s} {'avg_R':>8s} {'win%':>6s} {'PF':>6s} {'t_stat':>8s}")
    for row in ranked[:top_n]:
        s = row["stats"]
        print(f"{row['label']:30s} {s['n']:6d} {s['avg_r']:8.3f} {100*s['win_rate']:5.1f}% {s['profit_factor']:6.2f} {s['t_stat']:8.2f}")
    return ranked


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    dfs = load_cache(timeframe)
    print(f"Loaded {len(dfs)} cached symbols: {sorted(dfs)}")

    # --- Phase 1: fixed TP x SL grid ---
    phase1_rows = []
    for sl_mult in SL_MULTS:
        for tp_mult in TP_STDEV_MULTS:
            stats = el.run_pooled(dfs, sl_mult, el.tp_stdev(tp_mult))
            phase1_rows.append({"label": f"SL={sl_mult}x TP={tp_mult}x", "sl_mult": sl_mult, "tp_mult": tp_mult, "stats": stats})
    ranked1 = print_leaderboard("Phase 1: fixed TP x SL (stdev multiples)", phase1_rows, top_n=15)
    best_fixed = ranked1[0]
    best_sl = best_fixed["sl_mult"]
    print(f"\n--> best SL from phase 1: {best_sl}x (used as the fixed SL for phases 2-3)")

    # --- Phase 2: MA-touch TP ---
    phase2_rows = []
    for period in el.MA_PERIODS:
        for kind, col in [("EMA", f"ema_{period}"), ("VWAP", f"vwap_{period}")]:
            stats = el.run_pooled(dfs, best_sl, el.tp_ma(col))
            phase2_rows.append({"label": f"TP={kind}{period} SL={best_sl}x", "stats": stats})
    ranked2 = print_leaderboard("Phase 2: TP at MA touch (EMA / rolling VWAP)", phase2_rows, top_n=10)

    # --- Phase 3: indicator-level TP ---
    phase3_rows = []
    for level in RSI_LEVELS:
        stats = el.run_pooled(dfs, best_sl, el.tp_rsi(level))
        phase3_rows.append({"label": f"TP=RSI>={level} SL={best_sl}x", "stats": stats})
    for level in STOCH_LEVELS:
        stats = el.run_pooled(dfs, best_sl, el.tp_stoch(level))
        phase3_rows.append({"label": f"TP=Stoch>={level} SL={best_sl}x", "stats": stats})
    for num_std in BB_NUM_STD:
        stats = el.run_pooled(dfs, best_sl, el.tp_bb(num_std))
        phase3_rows.append({"label": f"TP=BB({num_std}std) SL={best_sl}x", "stats": stats})
    for mult in KC_MULTS:
        stats = el.run_pooled(dfs, best_sl, el.tp_kc(mult))
        phase3_rows.append({"label": f"TP=KC({mult}x) SL={best_sl}x", "stats": stats})
    ranked3 = print_leaderboard("Phase 3: TP at RSI / Stochastic / BB / KC levels", phase3_rows, top_n=20)

    # --- Overall leaderboard across all phases ---
    all_rows = phase1_rows + phase2_rows + phase3_rows
    print_leaderboard("OVERALL (all phases combined)", all_rows, top_n=15)


if __name__ == "__main__":
    main()
