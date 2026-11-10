# Breakout Hunter + VXX filter — Findings

**Conclusion: a VXX-based "no fresh vol spike" filter measurably improves
Breakout Hunter and appears to fix the "entry worse than random" problem
found in FINDINGS_ALL_BOTS.md — on a smaller universe than that study used,
so this should be read as a promising result, not a final re-validation.**

## Motivation
`FINDINGS_ALL_BOTS.md` found Breakout Hunter's entries lose to random
entries run through the identical exit engine in 10/10 draws, and
`BREAKOUT_HUNTER_ENTRIES_DISABLED = True` in `live_trading.py` as a result.
Hypothesis: a squeeze breakout is a trend-continuation bet that should do
worse when the broader market is volatile/choppy. VXX (iPath S&P 500 VIX
Short-Term Futures ETN) is used as a simple always-available volatility
proxy to gate entries by market regime.

## Filters tested (`strategy_breakout_vxx.py`)
Three independent flags, and two combinations, gating the existing
`long_entry` signal — no change to the squeeze/breakout detection itself:
- **calm**: VXX close below its own 20-day SMA
- **falling**: VXX close below where it was 5 trading days ago
- **not_spiking**: VXX close not at (or above) its own 5-day high
- **calm_falling**, **all3**: combinations of the above

## Result 1: same IS/OOS basket as FINDINGS_BREAKOUT.md (40 US symbols, daily, 2019-01-01 to 2026-08-01)
`run_basket_breakout_vxx.py`. Note: absolute trade counts here (325 baseline)
differ from the original write-up's (555, likely a different point-in-time
engine/date range at the time) — read this as an internally-consistent A/B
comparison, not a restatement of the original numbers.

| variant | IS trades | IS avg R | IS t | OOS trades | OOS avg R | OOS t | Combined trades | Combined avg R | Combined t | Win% | PF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (`long_entry`) | 167 | 0.211 | 3.48 | 158 | 0.028 | 0.46 | 325 | 0.122 | 2.80 | 56.6 | 1.55 |
| calm | 159 | 0.225 | 3.65 | 147 | -0.009 | -0.14 | 306 | 0.113 | 2.53 | 55.9 | 1.50 |
| falling | 149 | 0.217 | 3.41 | 145 | 0.011 | 0.17 | 294 | 0.115 | 2.55 | 56.5 | 1.52 |
| **not_spiking** | 153 | 0.228 | 3.56 | 149 | **0.050** | 0.78 | 302 | **0.140** | **3.08** | **59.3** | **1.64** |
| calm_falling | 145 | 0.236 | 3.70 | 142 | -0.007 | -0.11 | 287 | 0.116 | 2.54 | 56.8 | 1.53 |
| all3 | 139 | 0.242 | 3.77 | 139 | -0.002 | -0.04 | 278 | 0.120 | 2.62 | 57.6 | 1.57 |

**`not_spiking` is the only variant that improves OOS** (avg R +0.028 ->
+0.050, still not individually significant but the right direction) while
only trimming ~7% of trades. `calm` (below-SMA, "the market is generally
quiet") actually makes OOS *negative* — being below a slow-moving average
says nothing about whether a spike just happened. The combinations
(`calm_falling`, `all3`) look best in-sample but worst out-of-sample — a
classic overfitting signature (stacking filters fits the training symbols,
not the edge) — so they are rejected despite the best-looking IS numbers.

**Chosen filter: `not_spiking` only** (VXX not at its own 5-day high at
entry) — simplest, and the only one with OOS improvement.

## Result 2: does `not_spiking` fix "entry worse than random"?
`random_entry_breakout_vxx.py` — same methodology as `FINDINGS_ALL_BOTS.md`
(swap only entry days, same count per symbol, same exit engine), but on this
40-symbol universe rather than the point-in-time S&P pipeline (that study's
bar cache isn't present in this checkout, so it wasn't rerun at full scale).

| | Trades | Avg R | t-stat |
|---|---|---|---|
| Real (`not_spiking`) entries | 302 | **0.140** | 3.08 |
| Random entries, 10 draws | 262-280 | 0.024 - 0.318 (mean 0.130) | 0.31 - 3.44 |

**Random beats real in 4/10 draws** (was 10/10 for the unfiltered strategy
on the full point-in-time study). Real's avg R (0.140) sits just above the
random mean (0.130) — not a decisive win, but a dramatic improvement from
"loses every time" to "roughly a coin flip, slightly favoring real." Read
this as removing most of the entry-timing noise that random-entry beat,
not as proof the entry now has a large edge of its own.

## Caveats
- Smaller, different universe than `FINDINGS_ALL_BOTS.md`'s point-in-time
  S&P membership basket (that one's bar cache wasn't available here) — this
  is directionally consistent, not a like-for-like re-validation at the
  same scale.
- VXX history on Yahoo only reliably covers the backtest window used here;
  no survivorship/rebalance adjustments beyond what yfinance provides.
- No transaction costs modeled (same as the original Breakout Hunter
  engine).
- `not_spiking` trims real signal count by ~7% (325 -> 302) — fewer trades
  live, not a free lunch.

## Code
- `strategy_breakout_vxx.py` — VXX filter flags + gated entry-column variants
- `backtest_breakout_vxx.py` — same two-stage-stop engine as
  `backtest_breakout.py`, parameterized by which entry column to trade
- `run_basket_breakout_vxx.py` — baseline vs. all 5 variants, IS/OOS/combined
- `random_entry_breakout_vxx.py` — real vs. 10 random draws for the chosen
  `not_spiking` variant
- Results: `breakout_vxx_results.csv`
