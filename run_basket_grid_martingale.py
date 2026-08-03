"""Grid-Martingale basket backtest, US symbols only (v1 scope -- the
squeeze+breakout leg-0 signal this strategy reuses has no demonstrated edge
on crypto anywhere in this project, so crypto is deferred to a follow-up
rather than included on day one). Usage:
    python run_basket_grid_martingale.py

Long only, MIN_BARS_BETWEEN_LEGS=1 (both now the module defaults in
backtest_grid_martingale.py) -- short showed no support in the original
in-sample/out-of-sample check (see FINDINGS_GRID_MARTINGALE.md), and shrinking
the leg-add cooldown improved both win rate and avg R in-sample without hurting
out-of-sample (see sweep_grid_legtiming.py / the "Result 1b" update in the
same FINDINGS file for the honest caveat: the improvement is concentrated
in-sample, since out-of-sample symbols rarely engage the multi-leg mechanic
at all regardless of the cooldown).

One BetaBinomialPosterior is shared across every symbol in the basket (passed
through basket.run_basket's backtest_kwargs, mutated in place across the
per-symbol sequential loop) -- pooled, not per-symbol, same convention as
every other basket run in this project.
"""
import backtest_grid_martingale as bt
from basket import US_SYMBOLS, run_basket
from bayesian_tracker import BetaBinomialPosterior
from data import fetch_yfinance

START = "2019-01-01"
END = "2026-08-01"


def main():
    print("=== Grid-Martingale, daily bars, US symbols, long-only, gap=1 (v1.1) ===\n")
    tracker = BetaBinomialPosterior()
    all_trades, by_group, stats = run_basket(
        fetch_fn=lambda symbol: fetch_yfinance(symbol, START, END),
        symbols_by_group=(("US", US_SYMBOLS),),
        backtest_fn=bt.run_backtest,
        backtest_kwargs={"posterior": tracker},
    )

    print(f"\nFinal pooled posterior: alpha={tracker.alpha:.1f} beta={tracker.beta:.1f} "
          f"mean={tracker.mean:.3f} n_updates={tracker.n_updates}")

    print("\n=== Leg-count distribution ===")
    closed = [t for t in all_trades if t.exit_price is not None]
    leg_dist = {}
    for t in closed:
        leg_dist[t.num_legs] = leg_dist.get(t.num_legs, 0) + 1
    print(f"  {dict(sorted(leg_dist.items()))}")

    return all_trades, by_group, stats


if __name__ == "__main__":
    main()
