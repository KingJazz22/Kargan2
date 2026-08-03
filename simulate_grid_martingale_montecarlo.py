"""Bootstrap tail-risk simulation for Grid-Martingale.

A single historical backtest run only samples ONE ordering of wins/losses --
for a strategy with escalating (martingale-style) position sizing, sequence
matters a lot more than it does for a fixed-fractional strategy, so one path
badly understates tail/ruin risk. This resamples the pool of closed-trade
R-multiples (with replacement, order randomized) many times and compounds
each resampled sequence through a fixed-fractional equity model (equity *=
1 + risk_pct * R per trade, risk_pct = basket.py's RISK_PCT = 0.01, the same
sizing rule that actually generated these R-multiples) to get a DISTRIBUTION
of terminal equity / max drawdown / ruin probability, not just one point
estimate.

Simplification (flagged, not hidden): this compounds trades as one
sequential stream, not a full concurrent-multi-position multi-symbol
portfolio replay -- consistent with this project's existing "research-grade,
not execution-grade" disclaimers (see backtest.py's own docstring). It still
directly answers the tail-risk question the martingale sizing specifically
raises: how bad can a bad ordering of the SAME observed trade outcomes be.

Uses long-only, combined in-sample + out-of-sample trades (see
test_grid_oos_symbols.py) -- short showed no support in either sample, so
running the Monte Carlo on a strategy config that isn't actually being
recommended wouldn't be a meaningful use of this analysis.
"""
import numpy as np

import backtest_grid_martingale as bt
from basket import US_SYMBOLS, RISK_PCT, run_basket
from bayesian_tracker import BetaBinomialPosterior
from data import fetch_yfinance
from test_mtf_oos_symbols import OOS_US_SYMBOLS

START = "2019-01-01"
END = "2026-08-01"
N_RESAMPLES = 2000
DD_RUIN_THRESHOLD = -0.30
STARTING_EQUITY = 100_000.0


def compounded_equity_curve(r_multiples, risk_pct=RISK_PCT, starting_equity=STARTING_EQUITY):
    equity = starting_equity
    curve = [equity]
    for r in r_multiples:
        equity *= (1 + risk_pct * r)
        curve.append(equity)
    return np.array(curve)


def max_drawdown(curve):
    running_max = np.maximum.accumulate(curve)
    dd = curve / running_max - 1
    return dd.min()


def bootstrap(r_multiples, n_resamples=N_RESAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    n = len(r_multiples)
    terminal_equities, max_dds, ruined = [], [], []
    for _ in range(n_resamples):
        sample = rng.choice(r_multiples, size=n, replace=True)
        curve = compounded_equity_curve(sample)
        terminal_equities.append(curve[-1])
        dd = max_drawdown(curve)
        max_dds.append(dd)
        ruined.append(dd <= DD_RUIN_THRESHOLD)
    return np.array(terminal_equities), np.array(max_dds), np.array(ruined)


def main():
    fetch_fn = lambda symbol: fetch_yfinance(symbol, START, END)
    print("=== Gathering combined (in-sample + OOS) long-only trades ===")
    in_sample, _, _ = run_basket(
        fetch_fn=fetch_fn, symbols_by_group=(("US", US_SYMBOLS),),
        backtest_fn=bt.run_backtest, backtest_kwargs={"posterior": BetaBinomialPosterior()}, verbose=False,
    )
    oos, _, _ = run_basket(
        fetch_fn=fetch_fn, symbols_by_group=(("US", OOS_US_SYMBOLS),),
        backtest_fn=bt.run_backtest, backtest_kwargs={"posterior": BetaBinomialPosterior()}, verbose=False,
    )
    all_trades = [t for t in (in_sample + oos) if t.exit_price is not None and t.side == "long"]
    r_multiples = np.array([t.r_multiple() for t in all_trades])
    print(f"n={len(r_multiples)} long trades, avg_R={r_multiples.mean():.3f}, risk_pct={RISK_PCT}")

    print("\n=== Reference: actual chronological order (as observed) ===")
    chrono_curve = compounded_equity_curve(r_multiples)
    print(f"  terminal equity: ${chrono_curve[-1]:,.0f}  ({100*(chrono_curve[-1]/STARTING_EQUITY-1):+.1f}%)")
    print(f"  max drawdown: {100*max_drawdown(chrono_curve):.1f}%")

    print(f"\n=== Bootstrap: {N_RESAMPLES} resampled orderings ===")
    terminal_equities, max_dds, ruined = bootstrap(r_multiples)
    pct_return = 100 * (terminal_equities / STARTING_EQUITY - 1)

    print(f"  terminal equity  mean=${terminal_equities.mean():,.0f}  median=${np.median(terminal_equities):,.0f}")
    print(f"  terminal return  5th pct={np.percentile(pct_return,5):+.1f}%  "
          f"median={np.percentile(pct_return,50):+.1f}%  95th pct={np.percentile(pct_return,95):+.1f}%")
    print(f"  max drawdown     5th pct={100*np.percentile(max_dds,5):.1f}%  "
          f"median={100*np.percentile(max_dds,50):.1f}%  95th pct={100*np.percentile(max_dds,95):.1f}%")
    print(f"  P(drawdown <= {100*DD_RUIN_THRESHOLD:.0f}%): {100*ruined.mean():.1f}%")
    print(f"  P(terminal return < 0): {100*np.mean(pct_return < 0):.1f}%")


if __name__ == "__main__":
    main()
