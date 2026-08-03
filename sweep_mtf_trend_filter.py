"""Sweeps MTF v2's trend filter across 3 filter types (SMA/EMA/VWAP) x 4
lookback periods (50/100/150/200 days) = 12 variants, plus the no-filter
baseline. Guards against the exact overfitting trap that bit the
mean-reversion 30m parameter sweep earlier in this project: variants are
ranked using ONLY US_SYMBOLS (in-sample, 20 symbols). The winner (and
runner-up) are then re-tested on OOS_US_SYMBOLS -- 20 fresh symbols never
used to pick anything -- before being trusted. If a variant's edge doesn't
survive that OOS check, it's reported as likely overfit, not as a result.

Raw daily/4H data is fetched once (Alpaca, ~10.5yr) and cached across all 13
variants -- only the trend-line computation and backtest replay differ per
variant, so this doesn't re-hit the API 13x.
"""
import numpy as np

import simulate_live_system as sim
from basket import US_SYMBOLS
from test_mtf_oos_symbols import OOS_US_SYMBOLS

FILTER_TYPES = ["sma", "ema", "vwap"]
PERIODS = [50, 100, 150, 200]


def mtf_stats(closed_trades):
    trades = [t for t in closed_trades if t.strategy == "mtf"]
    n = len(trades)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "t_stat": 0.0, "pnl": 0.0}
    r = np.array([t.r_multiple() for t in trades])
    win_rate = float((r > 0).mean())
    avg_r = float(r.mean())
    std_r = float(r.std(ddof=1)) if n > 1 else 0.0
    t_stat = avg_r / (std_r / np.sqrt(n)) if std_r > 0 else 0.0
    pnl = float(sum(t.pnl() for t in trades))
    return {"n": n, "win_rate": win_rate, "avg_r": avg_r, "t_stat": t_stat, "pnl": pnl}


def run_variant(symbols, mtf_filter, raw_cache):
    trades, _ = sim.run_combined(symbols, sim.STARTING_EQUITY, sim.RISK_PCT,
                                  sim.MAX_CONCURRENT_POSITIONS, verbose=False,
                                  mtf_trend_filter=mtf_filter, raw_cache=raw_cache)
    return mtf_stats(trades)


def fmt_row(label, s):
    return (f"{label:<16}{s['n']:>6}{s['win_rate']*100:>8.1f}%{s['avg_r']:>+9.3f}"
            f"{s['t_stat']:>+8.2f}{s['pnl']:>+12,.2f}")


if __name__ == "__main__":
    print("Fetching raw data for all 40 symbols (cached across all variants)...")
    raw_cache = sim.fetch_raw(US_SYMBOLS + OOS_US_SYMBOLS)

    variants = [("none", 0)] + [(t, p) for t in FILTER_TYPES for p in PERIODS]

    print(f"\n{'='*70}\nIN-SAMPLE SWEEP (US_SYMBOLS, 20 symbols) -- picking a winner\n{'='*70}")
    header = f"{'variant':<16}{'n':>6}{'win%':>9}{'avg R':>9}{'t-stat':>8}{'PnL':>12}"
    print(header); print("-" * len(header))

    is_results = {}
    for ftype, period in variants:
        label = "baseline" if ftype == "none" else f"{ftype}_{period}"
        mtf_filter = None if ftype == "none" else (ftype, period)
        s = run_variant(US_SYMBOLS, mtf_filter, raw_cache)
        is_results[label] = (mtf_filter, s)
        print(fmt_row(label, s))

    ranked = sorted(is_results.items(), key=lambda kv: -kv[1][1]["t_stat"])
    print("\nRanked by in-sample t-stat (highest first):")
    for label, (_, s) in ranked[:5]:
        print(f"  {label}: t={s['t_stat']:+.2f}, avg_R={s['avg_r']:+.3f}, n={s['n']}")

    top2 = [r for r in ranked if r[0] != "baseline"][:2]

    print(f"\n{'='*70}\nOUT-OF-SAMPLE CONFIRMATION (OOS_US_SYMBOLS, 20 fresh symbols)\n{'='*70}")
    print(header); print("-" * len(header))

    baseline_oos = run_variant(OOS_US_SYMBOLS, None, raw_cache)
    print(fmt_row("baseline", baseline_oos))

    oos_results = {}
    for label, (mtf_filter, is_s) in top2:
        s = run_variant(OOS_US_SYMBOLS, mtf_filter, raw_cache)
        oos_results[label] = s
        print(fmt_row(label, s))

    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    for label, (_, is_s) in top2:
        oos_s = oos_results[label]
        holds = oos_s["t_stat"] > 0.5 and oos_s["avg_r"] > 0
        print(f"\n{label}: in-sample t={is_s['t_stat']:+.2f} (avg_R={is_s['avg_r']:+.3f}, n={is_s['n']}) "
              f"-> out-of-sample t={oos_s['t_stat']:+.2f} (avg_R={oos_s['avg_r']:+.3f}, n={oos_s['n']})")
        if holds:
            print("  HOLDS UP out of sample -- plausible real effect, not just in-sample noise.")
        else:
            print("  DOES NOT hold up out of sample -- likely overfit to the 20 in-sample symbols.")
