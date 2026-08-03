"""Final confirmation of PCSE's best-found exit rule, run across the full
27-symbol basket, plus a first-half/second-half split-sample check (the exit
parameters were chosen by looking at the full period's pooled stats, so this
is a robustness check against "all the edge lives in one era," not a true
held-out test).

Best found (see FINDINGS_CANDLE_PROB.md "Exit Strategy Search"):
  SL = 12x entry-time return-stdev (fixed, not trailing)
  TP = price touches the 20-bar Bollinger upper band (mid + 3.0 x std)

Usage: python run_exit_final.py
"""
import exit_lab as el
from run_exit_sweep import load_cache, print_leaderboard

BEST_SL_MULT = 12.0
BEST_TP_RULE = el.tp_bb(3.0)
BEST_LABEL = "SL=12x TP=BB(3.0std)"


def main():
    dfs = load_cache("1h")
    print(f"Loaded {len(dfs)} symbols: {sorted(dfs)}\n")

    configs = [
        (BEST_LABEL, BEST_SL_MULT, BEST_TP_RULE),
        ("SL=10x TP=BB(3.0std)", 10.0, el.tp_bb(3.0)),
        ("SL=12x TP=RSI>=80", 12.0, el.tp_rsi(80)),
        ("SL=20x TP=RSI>=80", 20.0, el.tp_rsi(80)),
        ("SL=15x TP=RSI>=85", 15.0, el.tp_rsi(85)),
    ]
    rows = [{"label": label, "stats": el.run_pooled(dfs, sl, tp)} for label, sl, tp in configs]
    print_leaderboard("FINAL: full 27-symbol basket, best exit configs", rows, top_n=10)

    print(f"\n=== Per-symbol breakdown: {BEST_LABEL} ===")
    for symbol, prep in sorted(dfs.items()):
        rows_s = prep.iloc[el.WARMUP_BARS:]
        if len(rows_s) < 10:
            continue
        trades, equity = el.simulate_exit(rows_s, BEST_SL_MULT, BEST_TP_RULE)
        s = el.pooled_stats(trades)
        final_eq = equity["equity"].iloc[-1] if len(equity) else 100_000.0
        ret_pct = (final_eq / 100_000.0 - 1) * 100
        print(f"  {symbol:10s} n={s['n']:4d} avg_R={s['avg_r']:7.3f} win%={100*s['win_rate']:5.1f} "
              f"PF={s['profit_factor']:5.2f} t={s['t_stat']:6.2f} return={ret_pct:+7.2f}%")

    print(f"\n=== Split-sample stability check: {BEST_LABEL} ===")
    first_half_trades, second_half_trades = [], []
    for symbol, prep in dfs.items():
        rows_s = prep.iloc[el.WARMUP_BARS:]
        if len(rows_s) < 20:
            continue
        mid = len(rows_s) // 2
        t1, _ = el.simulate_exit(rows_s.iloc[:mid], BEST_SL_MULT, BEST_TP_RULE)
        t2, _ = el.simulate_exit(rows_s.iloc[mid:], BEST_SL_MULT, BEST_TP_RULE)
        first_half_trades.extend(t1)
        second_half_trades.extend(t2)
    print("  first half (earlier history):", el.pooled_stats(first_half_trades))
    print("  second half (later history): ", el.pooled_stats(second_half_trades))


if __name__ == "__main__":
    main()
