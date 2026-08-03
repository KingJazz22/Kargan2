"""Parameter sweep: MIN_BARS_BETWEEN_LEGS, long-only.

Motivation: FINDINGS_GRID_MARTINGALE.md found win rate was already strong
(71-73%) but avg R modest, because 92% of trades resolve in a single leg
before the martingale averaging mechanic ever gets a chance to engage (the
0.75x-ATR take-profit sits closer than the 1.0x-ATR leg-step trigger). The
leg-timing gate (MIN_BARS_BETWEEN_LEGS) sits in front of that trigger check
entirely -- while the gate is closed, a quick dip-then-recover before it
reopens means the leg-add opportunity is missed outright, not just delayed.
A shorter gap lets more of those quick dips actually get averaged into
before price recovers past the trigger level, which should let more grids
capture the wider profit a multi-leg weighted-average produces.

Run on the in-sample 20-symbol US basket only (fast iteration); the winning
config still needs its own out-of-sample check (test_grid_oos_symbols.py)
before being trusted -- this script is for picking a candidate, not validating
one.
"""
import numpy as np

import backtest_grid_martingale as bt
from basket import US_SYMBOLS, run_basket
from bayesian_tracker import BetaBinomialPosterior
from data import fetch_yfinance

START = "2019-01-01"
END = "2026-08-01"
SWEEP = [0, 1, 2, 3, 5, 8]


def main():
    fetch_fn = lambda symbol: fetch_yfinance(symbol, START, END)
    print(f"{'gap':>4}  {'n':>5}  {'win%':>6}  {'avgR':>7}  {'t':>6}  {'return%':>8}  {'multi-leg%':>10}")

    for gap in SWEEP:
        tracker = BetaBinomialPosterior()
        trades, by_group, stats = run_basket(
            fetch_fn=fetch_fn,
            symbols_by_group=(("US", US_SYMBOLS),),
            backtest_fn=bt.run_backtest,
            backtest_kwargs={"posterior": tracker, "long_only": True, "min_bars_between_legs": gap},
            verbose=False,
        )
        closed = [t for t in trades if t.exit_price is not None]
        if not closed:
            print(f"{gap:>4}  no trades")
            continue
        multi_leg_pct = 100 * sum(1 for t in closed if t.num_legs >= 2) / len(closed)
        total_return_pct = sum(by_group["return_pct"])  # sum of per-symbol standalone returns, rough profit proxy
        print(f"{gap:>4}  {stats['num_trades']:>5}  {stats['win_rate_pct']:>6.1f}  "
              f"{stats['avg_r_multiple']:>7.3f}  {stats['t_stat_vs_zero']:>6.2f}  "
              f"{total_return_pct:>8.1f}  {multi_leg_pct:>9.1f}%")


if __name__ == "__main__":
    main()
