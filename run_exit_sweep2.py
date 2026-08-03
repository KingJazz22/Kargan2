"""Follow-up exit sweep: phase 1 found SL=4.0x (the widest tested) as best,
suggesting the true optimum is wider still. Extend the SL range against the
top TP rules from the first sweep, then test combining the top TP rules
(whichever fires first) at the best SL found.
"""
import glob
import os

import pandas as pd

import exit_lab as el
from run_exit_sweep import load_cache, print_leaderboard

SL_MULTS_WIDE = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
TOP_TPS = [
    ("RSI>=80", el.tp_rsi(80)),
    ("RSI>=75", el.tp_rsi(75)),
    ("BB(3.0std)", el.tp_bb(3.0)),
    ("BB(2.5std)", el.tp_bb(2.5)),
    ("EMA200", el.tp_ma("ema_200")),
]


def tp_combine(rules):
    """Exit at whichever of several PRICE-kind TP rules gives the LOWEST
    (soonest-reached) target this bar -- "take profit at the first of these
    levels touched." All rules here happen to be "price" kind."""
    fns = [fn for kind, fn in rules]

    def combined(entry_price, ret_stdev, row):
        targets = [f(entry_price, ret_stdev, row) for f in fns]
        targets = [t for t in targets if t is not None]
        return min(targets) if targets else None
    return ("price", combined)


def main():
    dfs = load_cache("1h")
    print(f"Loaded {len(dfs)} cached symbols: {sorted(dfs)}")

    # --- extend SL range against the strongest TPs ---
    rows = []
    for sl_mult in SL_MULTS_WIDE:
        for tp_name, tp_rule in TOP_TPS:
            stats = el.run_pooled(dfs, sl_mult, tp_rule)
            rows.append({"label": f"SL={sl_mult}x TP={tp_name}", "sl_mult": sl_mult, "tp_name": tp_name, "stats": stats})
    ranked = print_leaderboard("Extended SL range x top TP rules", rows, top_n=20)
    best = ranked[0]
    print(f"\n--> best overall: {best['label']}  t={best['stats']['t_stat']:.2f}")

    # --- combine top TP rules at the best SL found ---
    # tp_rsi/tp_stoch are "signal" kind (checked on close), not a price
    # target, so they can't be combined via min() with price-kind rules --
    # combine only the price-kind rules (BB, EMA) here.
    best_sl = best["sl_mult"]
    price_combo_rows = []
    price_candidates = [
        ("BB3.0", el.tp_bb(3.0)), ("BB2.5", el.tp_bb(2.5)), ("EMA200", el.tp_ma("ema_200")),
    ]
    combo_ab = tp_combine([price_candidates[0][1], price_candidates[1][1]])
    stats = el.run_pooled(dfs, best_sl, combo_ab)
    price_combo_rows.append({"label": f"SL={best_sl}x TP=min(BB3.0,BB2.5)", "stats": stats})

    combo_all = tp_combine([p[1] for p in price_candidates])
    stats = el.run_pooled(dfs, best_sl, combo_all)
    price_combo_rows.append({"label": f"SL={best_sl}x TP=min(BB3.0,BB2.5,EMA200)", "stats": stats})

    print_leaderboard("Combined price-based TP rules at best SL", price_combo_rows, top_n=10)


if __name__ == "__main__":
    main()
