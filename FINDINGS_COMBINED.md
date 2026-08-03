# Combined Portfolio (Breakout Hunter + Multi-Timeframe RSI+Stoch) — finding

**Conclusion: running both validated strategies together from ONE shared
capital pool produces a modest, unspectacular portfolio result (+38.9% over
~2yr, Sharpe ~0.86, max DD -13.2%) -- and importantly, capital crowding
dilutes both strategies' individual edges rather than simply summing them.
MTF's edge, significant on its own (t=2.56), drops below significance when
sharing capital with Breakout Hunter (t=1.46).**

## Scope
Only the two strategies in this project with a demonstrated edge: Breakout
Hunter (daily, US) and Multi-Timeframe RSI+Stochastic long-only (1H+4H, US).
Trend Following and Mean Reversion were excluded -- neither showed an edge
alone, so including them wasn't a meaningful test.

One shared capital pool ($100k starting), each new entry (either strategy)
sized off CURRENT total equity x 1% risk / stop distance -- same approach as
`live_breakout.py`. All 40 validated US symbols (`US_SYMBOLS + OOS_US_SYMBOLS`).
A symbol can hold positions in both strategies simultaneously (tracked as
separate lots), same as two independent bots trading one brokerage account.

## Engine
`backtest_combined.py` -- a new mixed-timeframe event loop (master clock =
union of all 1H timestamps), reusing each strategy's exact validated
signal/stop logic (both are price/ATR-driven, independent of account size).
Breakout Hunter's daily logic fires once per day at the first 1H tick, using
the just-completed prior daily bar -- mirrors `backtest_breakout.py` exactly.
MTF's hourly logic is unchanged from `backtest_mtf.py`.

## Bug found and fixed before trusting the result
Initial run showed suspiciously strong numbers (+155% return, Sharpe 1.2).
Position sizing capped share count by `equity / entry_price` (total mark-to-
market equity) instead of `cash / entry_price` (actually available cash).
With multiple simultaneous positions across 2 strategies x 40 symbols, this
let the engine implicitly spend more cash than it had -- unmodeled leverage.
Fixed by capping with available cash instead (risk_amount still sizes off
total equity, which is correct/intentional; only the no-leverage cap needed
to use cash).

| | Before fix (buggy) | After fix (correct) |
|---|---|---|
| Trades | 908 | 524 |
| Total return | +155.2% | +38.9% |
| Max drawdown | -34.9% | -13.2% |
| Sharpe (approx) | 1.2 | 0.86 |
| Breakout n / avg R / t | 95 / 0.094 / 1.07 | 69 / 0.073 / 0.70 |
| MTF n / avg R / t | 813 / 0.121 / 2.89 | 455 / 0.064 / 1.46 |

All numbers below are post-fix.

## Result: capital crowding, not additive edges
| | MTF standalone (40 symbols, own $100k) | MTF inside combined portfolio |
|---|---|---|
| Trades | 924 | 455 |
| Avg R | +0.087 | +0.064 |
| t-stat | 2.56 (significant) | 1.46 (not significant) |

Sharing one pool across 40 symbols and 2 strategies means there's often not
enough free cash to take every signal both strategies would have taken
independently -- trade count nearly halved, and the statistical confidence in
MTF's edge (which held up on its own) doesn't survive the competition for
capital. This is the central finding: running two validated strategies "at
the same time" doesn't combine their edges additively -- they compete for
capital, and that dilutes the realized result below what either showed alone.

## Position-count cap test
Tested `max_concurrent_positions` at 3, 5, 10, 20, and uncapped:

| Cap | Trades | Return | Sharpe | Breakout avg R / t | MTF avg R / t |
|---|---|---|---|---|---|
| 3 | 338 | +19.9% | 0.58 | -0.146 / -1.02 | +0.078 / 1.24 |
| 5 | 442 | +25.8% | 0.65 | -0.102 / -0.89 | +0.075 / 1.46 |
| 10 | 520 | +35.8% | 0.82 | +0.059 / 0.56 | +0.061 / 1.41 |
| 20 | 533 | +40.1% | 0.88 | +0.090 / 0.88 | +0.062 / 1.46 |
| Uncapped | 524 | +38.9% | 0.86 | +0.073 / 0.70 | +0.064 / 1.46 |

Cash was already the binding constraint -- caps of 10-20 barely differ from
uncapped, since the portfolio rarely holds that many positions at once anyway.
A tight cap (3-5) doesn't just shrink the sample, it actively hurts (Breakout
Hunter's avg R turns negative at cap=3): which signals get the scarce slots
is decided by processing order, not signal quality, so a tight cap excludes
trades somewhat arbitrarily rather than keeping the best ones. `live_breakout.py`'s
`MAX_CONCURRENT_POSITIONS = 10` sits in the non-binding range -- a reasonable
safety ceiling that shouldn't materially constrain normal operation, based on
this window.

## Caveats
- **Window**: capped at ~2 years by MTF's 1H data limit (Yahoo). Breakout
  Hunter's own validation used the full 6.5-year daily history -- this test
  says nothing about the pair's behavior across a full market cycle,
  including a real bear market.
- **Position-selection order under a cap is arbitrary** (alphabetical
  processing order, not signal quality) -- fine at loose caps where it rarely
  binds, but means a tight cap's results shouldn't be read as "the best
  trades survive," just "fewer trades, chosen somewhat at random."
- Same correlated-symbols caveat as every other basket test in this project.

## Code
- `backtest_combined.py` -- shared-capital, mixed-timeframe portfolio engine
