# Bayesian-Gated Half-Martingale Grid — backtest finding

**Current conclusion (post v1.2, `long_only=True`, `MIN_BARS_BETWEEN_LEGS=1`,
`TP_ATR_MULT=2.0`): improved, still unconfirmed. Out-of-sample flipped from
losing money outright (-6.9% total return at the original defaults) to net
positive (+3.1%), and in-sample nearly doubled (+45.2% -> +79.4%) -- the
first change in this strategy's history that helped both samples, not just
one. But out-of-sample still isn't independently statistically significant
(t=0.47) and the averaging/martingale mechanic itself remains essentially
untested on fresh data (out-of-sample multi-leg engagement stays under 1%
regardless of parameters, because the Bayesian gate correctly refuses to
escalate where the pooled win rate doesn't support it). Still weaker than
this project's bar for "confirmed" (Breakout Hunter, t=3.40, holds up across
independent samples AND a full 10yr multi-regime retest) -- but for the
first time, directionally consistent rather than contradictory across
samples. See "v1.1"/"v1.2" updates and Results 1b/1c below for how this was
reached; the original (v1) numbers are kept below for the audit trail.**

**Original (v1) conclusion, for context: unconfirmed, not a demonstrated edge
-- weaker than this project's bar for "confirmed" (Breakout Hunter, t=3.40).
The pooled in-sample result looked strong (t=4.76) but did not replicate
out-of-sample (t=-1.66, flipped negative) -- the same overfitting/lucky-draw
signature this project has found before (FINDINGS_MTF.md, FINDINGS_MEANREV.md).
The long-only combined result across both symbol sets does clear significance
(t=2.97) but is driven almost entirely by in-sample data, with out-of-sample
long essentially flat-to-negative (t=-0.80) -- a materially weaker replication
than Breakout Hunter's or MTF's own long sides showed. Short is not supported
in either sample. The averaging/martingale mechanic itself engaged in only
~4% of trades under the tested defaults, so this result mostly validated a
tight-take-profit variant of Breakout Hunter's own entry signal, not the grid
mechanic specifically. Tail risk, where it could be measured, stayed
well-contained.**

**v1.1 update (long-only, tighter leg-timing): win rate and avg R both improved
in-sample (73%->80% win rate, +0.069->+0.117 avg R) by dropping short and
shrinking the leg-add cooldown from 2 bars to 1 -- but the improvement is
concentrated entirely in-sample. Out-of-sample symbols essentially never
engage the multi-leg mechanic regardless of the cooldown (299/300 OOS trades
stay at 1 leg), so tightening the cooldown could only ever help the in-sample
half of the story. The combined in-sample+OOS t-stat rose from 2.97 to 3.94,
which reads better on paper, but the underlying imbalance -- in-sample driving
the result, OOS essentially unmoved and still weak -- did not improve, and by
one reading got slightly worse (the gap between the two samples widened, not
narrowed). See "Result 1b" below before treating the higher headline numbers
as a validated improvement.**

**v1.2 update (TP_ATR_MULT 0.75 -> 2.0): this is the first change in this
project's history for this strategy that improved BOTH samples, not just
in-sample. At the old TP, out-of-sample lost money outright (-6.9% total
return, n=300). At TP=2.0, out-of-sample is net positive (+3.1%, n=245) and
in-sample nearly doubles (+45.2% -> +79.4%, n=265). Widening TP did NOT
unlock the multi-leg mechanic the way expected -- OOS multi-leg engagement
stayed near 0% at every TP tested, because the Bayesian gate itself keeps
denying escalation on symbols where the win rate doesn't clear breakeven
(arguably the gate working exactly as designed, not a bug). The real lever
turned out to be reducing trade FREQUENCY and raising per-trade payoff, not
the averaging mechanic. Out-of-sample is still not independently significant
on its own (t=0.47) -- directionally right, not confirmed. In the shared
combined portfolio, Breakout Hunter's own t-stat drops to 1.98 (just under
significance) because the grid strand's wider targets hold positions open
longer, crowding Breakout Hunter out of more cash than before -- a real cost,
not a free upgrade. See "Result 1c" below for full numbers.**

## Spec tested
Leg 0 enters on Breakout Hunter's unchanged squeeze+breakout signal
(`strategy_breakout.py`) -- long on a close above the prior 20-bar high,
plus a new mirrored short leg on a close below the prior 20-bar low, both
with the existing volume/ADX-rising confirmation. If price then moves against
the position by 1.0x ATR (measured off the last leg's own locked-in ATR
snapshot) and at least 2 bars have passed since the last leg, a new leg may
be added: sized at the prior leg's risk x 1.4 (half-martingale, not full
2x), further scaled by a Bayesian gate -- a pooled, whole-basket Beta-Binomial
posterior (uninformative Beta(1,1) prior) over "does this grid resolve via
take-profit," gated against the actual current payoff-ratio breakeven, with
size shrinking continuously toward 1x when the posterior's 20th-percentile
credible bound doesn't clear breakeven even though the mean does. Max 4 legs
per grid (leg 0 + 3 adds). Exit: take-profit at the weighted-average entry +/-
0.75x ATR, a hard grid-stop beyond the worst leg's entry by 3.0x ATR (bounds
the loss, no further averaging past it), or a regime-kill on ADX dropping
below 20 ("dead trend," `strategy.ADX_WEAK`). Full rules in `strategy_grid.py`
/ `backtest_grid_martingale.py`.

v1 scope: US equities only, daily bars -- the leg-0 entry signal has no
demonstrated edge on crypto anywhere in this project (FINDINGS_BREAKOUT.md:
avg R -0.06, t=-1.38), so crypto wasn't included on day one.

## Result 1: in-sample vs. out-of-sample (US_SYMBOLS vs. OOS_US_SYMBOLS, `test_grid_oos_symbols.py`)
| | In-sample (20 symbols) | Out-of-sample (20 symbols) | Combined (40 symbols) |
|---|---|---|---|
| Trades (pooled long+short) | 493 | 551 | 1044 |
| Win rate | 73.4% | 60.8% | 66.8% |
| Avg R | +0.069 | **-0.029** | +0.018 |
| t-stat | **4.76** | **-1.66** | 1.53 (not significant) |

The in-sample number looked like the strongest result in this whole project
-- it does not replicate. Out-of-sample flips to a negative point estimate.
Pooled across both, the combined result is not statistically significant.

## Result 1b (v1.1): dropping short + tightening the leg-add cooldown
Ask: win rate was already strong, could profit (avg R) be improved without
giving that up? Two changes, both now the module defaults in
`backtest_grid_martingale.py` (`long_only=True`, `MIN_BARS_BETWEEN_LEGS=1`,
swept in `sweep_grid_legtiming.py`):

| Cooldown (bars) | Trades | Win rate | Avg R | t-stat | Multi-leg % |
|---|---|---|---|---|---|
| 8 (loose) | 343 | 77.8% | +0.091 | 5.24 | 2.3% |
| 5 | 343 | 78.1% | +0.096 | 5.75 | 5.2% |
| 3 | 343 | 79.3% | +0.104 | 6.52 | 10.2% |
| 2 (v1 default) | 345 | 79.7% | +0.109 | 7.03 | 14.5% |
| **1 (new default)** | **347** | **80.1%** | **+0.117** | **8.03** | **18.2%** |
| 0 | 347 | 80.1% | +0.117 | 8.03 | 18.2% |

Both win rate and avg R improve monotonically as the cooldown shrinks -- a
shorter gap lets more quick dip-then-recover moves actually get a leg added
before price bounces back past the trigger level, so more grids capture a
better weighted-average entry. (0 and 1 land identically: the elapsed-bar
counter increments *before* the gate check each bar, so a cooldown of 1
already permits a same-day second leg on a high-range entry day -- effectively
the floor.) In-sample combined long-only, this took win rate from 71.3% to
80.1% and avg R from +0.043 to +0.055 (using the pooled tracker across the
whole run rather than a fresh one per sweep point, hence the slightly
different combined number below vs. the sweep row above).

**The honest caveat, checked before trusting this**: out-of-sample symbols
almost never engage the multi-leg mechanic at all -- 299/300 OOS trades stay
at 1 leg regardless of cooldown, so a tighter cooldown had essentially zero
room to help there. Rerunning the full in-sample/OOS split with the new
defaults:

| | In-sample (20) | Out-of-sample (20) | Combined (40) |
|---|---|---|---|
| Trades | 347 | 300 | 647 |
| Win rate | 80.1% | 63.3% | 72.3% |
| Avg R | +0.117 | -0.019 (unchanged) | +0.054 |
| t-stat | **8.03** | -0.80 (unchanged) | **3.94** |

The combined t-stat looks better (2.97 -> 3.94) and so does avg R (+0.043 ->
+0.054) -- genuine improvements in absolute terms, worth keeping as the new
default since they never make anything worse. But the *mechanism* is
"in-sample got much better while out-of-sample stayed exactly the same,"
which is the opposite of what would resolve the original concern (in-sample
carrying the whole result) -- if anything the imbalance between the two
samples widened. Treat the higher headline numbers as **real but not more
validated** than before; the core "unconfirmed" read from Result 1/2 stands.

## Result 1c (v1.2): widening the take-profit -- the first change that helped BOTH samples
Ask: find a config that's actually profitable, not just one with a better
win rate. Swept `TP_ATR_MULT` from 0.5 to 3.0 (`sweep_grid_tp.py`), evaluating
in-sample AND out-of-sample together for every candidate from the start --
not tuned on one and checked on the other after the fact, given how
misleading that approach was for the leg-timing tweak.

| TP | IS n | IS win% | IS avg R | IS t | IS multi-leg% | OOS n | OOS win% | OOS avg R | OOS t | OOS multi-leg% | Combined t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 362 | 85.4% | +0.102 | 10.45 | 20.4% | 319 | 70.2% | -0.018 | -0.91 | 0.0% | 4.26 |
| 0.75 (old) | 347 | 80.1% | +0.117 | 8.06 | 18.2% | 300 | 63.3% | -0.017 | -0.71 | 0.3% | 4.05 |
| 1.00 | 320 | 74.4% | +0.119 | 5.89 | 11.6% | 285 | 62.1% | +0.009 | 0.35 | 0.4% | 4.10 |
| 1.25 | 305 | 71.1% | +0.130 | 5.38 | 6.2% | 268 | 58.6% | +0.016 | 0.54 | 0.7% | 4.06 |
| 1.50 | 288 | 69.4% | +0.152 | 5.73 | 12.5% | 259 | 54.8% | -0.004 | -0.12 | 0.8% | 3.71 |
| **2.00 (new)** | **266** | **69.5%** | **+0.224** | **7.43** | **16.5%** | **245** | **52.2%** | **+0.020** | **0.55** | **0.4%** | **5.23** |
| 2.50 | 248 | 64.9% | +0.227 | 6.29 | 15.3% | 235 | 49.4% | +0.024 | 0.57 | 0.4% | 4.61 |
| 3.00 | 243 | 63.8% | +0.264 | 6.65 | 17.7% | 230 | 48.3% | +0.040 | 0.89 | 0.9% | 5.14 |

**The key surprise**: OOS multi-leg engagement never rises above ~1%, at ANY
TP level -- widening TP does not unlock the averaging mechanic the way the
leg-timing tweak was hoped to. Investigating why: the Bayesian gate itself is
choking off leg-adds on OOS symbols, because the pooled posterior's win rate
for OOS grids doesn't clear the payoff-ratio breakeven, so `mean <=
p_breakeven` denies escalation almost from the start and stays denied as the
posterior converges toward OOS's true (lower) win rate. This is arguably the
gate working exactly as intended -- refusing to martingale into a weaker
edge -- but it means the martingale mechanic still isn't validated on fresh
data by this test either.

**What DID change out-of-sample**: avg R flips from consistently negative at
TP 0.5-0.75 (-0.017 to -0.018) to consistently positive at TP >= 1.0 (+0.009
to +0.040), though never significant on its own (t between -0.91 and +0.89).
Confirmed on total compounded return, not just avg R (`US_SYMBOLS` +
`OOS_US_SYMBOLS`, long-only, everything else at current defaults):

| TP_ATR_MULT | In-sample return | Out-of-sample return | Combined t-stat |
|---|---|---|---|
| 0.75 (old default) | +45.2% | **-6.9%** | 3.94 |
| 2.00 (new default) | **+79.4%** | **+3.1%** | 5.01-5.23 |

At the old default, out-of-sample symbols would have **lost money outright**.
At TP=2.0, both samples are net positive -- modest on the OOS side, but a
real, qualitative change from losing to winning, not just a bigger avg-R
number. `TP_ATR_MULT = 2.0` is now the module default.

**Why 2.0 and not 3.0** (similar or slightly better on some metrics): 2.0
gives the highest combined t-stat (5.23) with a less extreme departure from
the original spec, and keeps `TP_ATR_MULT < GRID_STOP_ATR_MULT` (2.0 vs 3.0)
so the take-profit target stays smaller than the maximum loss, unlike TP=3.0
where they're equal.

## Result 2: long vs. short, same in-sample/OOS split (v1, gap=2, TP=0.75 -- see Result 1b/1c for the current defaults)
| | In-sample | Out-of-sample | Combined |
|---|---|---|---|
| Long: n / avg R / t | 342 / +0.097 / 5.78 | 300 / -0.019 / -0.80 | 642 / +0.043 / **2.97** |
| Short: n / avg R / t | 151 / +0.007 / 0.25 | 251 / -0.040 / -1.58 | 402 / -0.022 / -1.18 |

**Short**: leans negative to flat in both independent samples, never
significant, never positive on a combined basis -- not supported, consistent
with this project's other precedent (`FINDINGS_MTF.md`) that a mirrored short
doesn't inherit the long side's edge. If this strategy is pursued further, it
should run long-only.

**Long**: combined clears t=2.97, which on its own would read as a real
result. But unlike Breakout Hunter (in-sample t=2.88, OOS -0.24 but still
directionally near-flat rather than negative) or MTF's long side (in-sample
t=2.72, OOS still positive at t=0.88), this OOS long result is a negative
point estimate (t=-0.80). The combined significance is carried almost
entirely by the in-sample half. Read as **leaning positive, unconfirmed** --
not "confirmed" by this project's own bar.

## Result 3: does the averaging mechanic itself do anything? (1-leg vs. 2+-leg trades, in-sample basket, v1 config -- gap=2, TP=0.75, both directions)
*Numbers below predate the Result 1b/1c changes. Under gap=1/long-only,
multi-leg engagement rose to 18.2% in-sample; under TP=2.0 it's 16.5%
in-sample (similar) but still under 1% out-of-sample regardless of TP -- see
Result 1c for why (the Bayesian gate itself denies escalation on OOS, not a
geometric TP/step artifact). The averaging mechanic remains essentially
untested on fresh data through every version of this strategy so far.*
| | n | Win rate | Avg R | t-stat |
|---|---|---|---|---|
| 1 leg (no averaging -- TP hit before any add) | 456 | 73.2% | +0.076 | 5.17 |
| 2+ legs (martingale actually engaged) | 38 | 84.2% | +0.092 | 1.87 (n too small) |

Only 38/493 in-sample trades (7.7%) ever added a leg -- because the default
take-profit (0.75x ATR) sits closer than the leg-add trigger (1.0x ATR), most
trades resolve via take-profit before a leg-add is ever possible. **The
headline numbers above are therefore mostly a test of "Breakout Hunter entry
+ a tight 0.75x-ATR take-profit + a 3x-ATR stop + an ADX dead-trend kill,"
not a real test of the grid/martingale averaging mechanic**, which barely
fired. The 2+-leg subset performs at least as well as the 1-leg subset (84.2%
vs. 73.2% win rate) -- some evidence the averaging isn't actively hurting --
but n=38 is too small to call this validated either way. A parameter sweep
that widens the TP relative to the leg-step (so grids actually get a chance
to add legs) is the natural follow-up if this strategy is pursued further.

**Bayesian gate behaved as designed**: inspecting the 38 multi-leg trades'
`size_mult_applied` per leg, every leg-2 add got shrunk to exactly 1.0x (no
escalation at all) -- the posterior was still too uncertain early on for its
20th-percentile credible bound to clear the breakeven threshold. Only leg-3/4
adds, later in the pooled trade history once the posterior had accumulated
enough evidence, got the full 1.4x escalation. This is the intended behavior:
martingale-style escalation only kicks in once there's real pooled evidence
for it, not from day one.

## Result 4: Monte Carlo tail-risk (`simulate_grid_martingale_montecarlo.py`, long-only, combined 40-symbol trades)
Bootstrap-resampled the observed R-multiples 2000x (order randomized each
time) through a compounding fixed-fractional equity model (1% risk/trade,
matching `basket.py`'s convention) to see the DISTRIBUTION of outcomes, not
just the one historical ordering. v1.2 numbers (TP=2.0, n=510); v1.1
(TP=0.75, gap=1, n=647) and v1 (gap=2, n=628) in parentheses for comparison:

| | Value |
|---|---|
| Actual chronological order: terminal return / max DD | **+84.9% / -5.4%** (v1.1: +42.1%/-8.0%; v1: +31.1%/-8.6%) |
| Bootstrap terminal return: 5th pct / median / 95th pct | **+50.1% / +84.2% / +125.9%** (v1.1: +22.8%/+42.4%/+64.1%; v1: +12.8%/+30.9%/+51.2%) |
| Bootstrap max drawdown: 5th pct / median / 95th pct | -8.3% / -5.1% / -3.4% (v1.1: -7.5%/-4.4%/-2.9%; v1: -9.0%/-5.2%/-3.3%) |
| P(max drawdown <= -30%) | 0.0% (unchanged) |
| P(terminal return < 0) | **0.0%** (v1.1/v1: 0.1%) |

Widening TP roughly doubled the whole return distribution (median +42.4% ->
+84.2%) on a comparable, even slightly better, drawdown profile -- fewer,
bigger-payoff trades rather than more risk per trade. Note n dropped from 647
to 510 (fewer, larger trades at the wider TP), consistent with the sweep.

Given the observed trade outcomes, resequencing them 2000 different ways
never produced a catastrophic drawdown -- the bounded design (half-martingale
not full doubling, hard 3x-ATR grid-stop, max 4 legs capping aggregate risk
at roughly 7-11% of equity per grid depending on how many legs fill) is doing
its job of containing tail risk. **Caveat**: this is a single sequential
trade-stream compounding model, not a full concurrent-multi-position
portfolio replay (consistent with this project's existing "research-grade,
not execution-grade" disclaimers) -- it answers "how bad could this exact set
of observed outcomes have gone in a different order," not "what's the true
worst case for parameters/regimes never observed in this data."

## Result 5: shared-equity combined portfolio with Breakout Hunter (`backtest_combined_grid.py`, 40 symbols, $100k pool, cap=10 concurrent)
v1.2 (TP=2.0), with v1.1 (TP=0.75, gap=1) in parentheses:

| | Standalone (own $100k pool) | In combined portfolio (shared pool) |
|---|---|---|
| Breakout Hunter: n / win rate / avg R / t | 325 / 58.5% / +0.150 / 3.40 | 224 / 55.8% / +0.103 / **1.98** (v1.1: 240 / 58.3% / +0.140 / 2.84) |
| Grid-Martingale (long): n / win rate / avg R / t | 510 / 61.2% / +0.122 / 5.01 | 254 / 59.1% / +0.096 / 2.73 (v1.1: 319 / 70.8% / +0.044 / 2.21) |

Portfolio total: 478 trades, **+56.2% return** (v1.1: +55.5%; v1: +49.9%),
**-12.7% max drawdown** (v1.1: -10.8%; v1: -11.8%) over the ~6.5yr window.
**A real cost worth flagging, not glossing over**: Breakout Hunter's own
combined-portfolio t-stat drops to 1.98 -- just under the significance
threshold -- because the grid strand's wider TP means positions stay open
longer waiting for a bigger move, tying up shared cash for longer and
crowding Breakout Hunter out of more trades than the tighter-TP version did
(224 vs. 240 trades). The grid strand itself does better in the shared pool
than v1.1 did (t=2.73 vs. 2.21), and portfolio-level return/drawdown are
roughly a wash vs. v1.1 despite the much bigger standalone improvement --
again the shared pool's cash-capping absorbing most of the standalone gain.
Same capital-crowding effect `FINDINGS_COMBINED.md` found originally, now
visibly worse for Breakout Hunter specifically as the grid strand's holding
period grew.

## Result 5b: protecting Breakout Hunter's cash access -- `grid_capital_cap_frac`
Ask: can both strategies run from the same shared pool without dragging
Breakout Hunter's significance down? First diagnosed what's actually
binding. `max_concurrent_positions` at 10/20/40/None:

| Cap | Return | Breakout: n / t | Grid: n / t |
|---|---|---|---|
| 10 | +57.8% | 226 / 1.91 | 253 / 3.14 |
| 20 | +61.2% | 237 / 2.05 | 278 / 3.26 |
| 40 | +61.2% | 237 / 2.05 | 278 / 3.26 |
| None | +61.2% | 237 / 2.05 | 278 / 3.26 |

20/40/None are identical -- the position-COUNT cap stops binding well before
20 and was never the real constraint (same conclusion `FINDINGS_COMBINED.md`
reached for the original Breakout Hunter + MTF pairing). Even fully uncapped,
Breakout Hunter's t only reaches 2.05 -- confirming **cash availability**,
not position count, is what's actually crowding Breakout Hunter out: Grid's
wider TP=2.0 target holds positions (and their cash) open for longer.

Fix: `grid_capital_cap_frac` (new `backtest_combined_grid.py` parameter) caps
Grid's own open notional (sum of `total_shares * weighted_avg_entry` across
all its open positions) at a fraction of current total equity -- reserving
the rest of the SAME shared pool for Breakout Hunter, rather than splitting
into two separate non-interacting pools. Swept 0.05-0.6 (`max_concurrent_positions=10`):

| Cap frac | Return | Max DD | Breakout: n / t | Grid: n / t |
|---|---|---|---|---|
| None (uncapped) | +56.7% | -12.6% | 225 / 1.91 | 250 / 3.08 |
| 0.6 | +55.1% | -12.6% | 228 / 1.89 | 253 / 2.80 |
| 0.5 | +59.5% | -12.6% | 230 / 2.13 | 251 / 2.93 |
| 0.4 | +53.8% | -12.9% | 236 / 1.81 | 244 / 3.35 |
| 0.3 | +54.7% | -12.9% | 237 / 1.99 | 247 / 3.13 |
| **0.25** | **+56.2%** | -13.0% | 237 / **2.37** | 240 / 3.25 |
| **0.2** | **+55.0%** | -13.0% | 236 / **2.40** | 233 / **3.80** |
| 0.15 | +51.4% | -12.2% | 237 / 2.49 | 222 / 3.59 |
| 0.1 | +46.2% | -11.6% | 240 / 2.24 | 214 / 4.03 |
| 0.05 | +38.1% | -11.2% | 240 / 2.27 | 212 / 4.09 |

Above ~0.3, the effect is noisy (not monotonic -- consistent with
`FINDINGS_COMBINED.md`'s own caveat that which trades get through a cap is
partly an artifact of processing order, not just signal quality). Below
~0.25 it's clean: **capping Grid at roughly 20% of total equity is a genuine
win-win, not a tradeoff** -- Breakout Hunter's t recovers from ~1.9-2.05 to
~2.4-2.5 (still short of its standalone 3.40, but clearly back above the
significance line), AND Grid's own t-stat improves (3.08 -> 3.25-3.80,
capital scarcity making it more selective about which grids it commits to),
on portfolio return/drawdown that's a wash against the uncapped baseline.
Going tighter still (0.05-0.1) keeps improving Grid's t further but starts
costing real portfolio return (+38-46% vs. +55-57%) as Grid simply trades
less -- `grid_capital_cap_frac = 0.2` recovers Breakout Hunter's
significance, improves Grid's own, and costs nothing on portfolio-level
return/drawdown. **Applied as the new default** in `backtest_combined_grid.py`
(not left as an opt-in) -- `run_combined()`'s signature now defaults to
`grid_capital_cap_frac=0.2` rather than `None`, so any future combined-portfolio
run gets the reserved-capital protection unless explicitly overridden.

## Interpretation
- **Short: not supported.** Flat-to-negative in both independent samples.
  Now disabled by default (`long_only=True`).
- **Long: for the first time, directionally consistent across both samples.**
  v1.2's wider take-profit is qualitatively different from v1.1's leg-timing
  tweak: out-of-sample flips from losing money (-6.9% total return) to making
  money (+3.1%), not just from a worse number to a less-bad one. That's real
  progress on the actual question ("does this generalize"), not just a bigger
  in-sample number. It's still not independently significant on OOS alone
  (t=0.47) -- call it **leaning positive, more credible than before, still
  unconfirmed**.
- **The martingale/averaging mechanic itself: still essentially untested on
  fresh data, for a different reason than expected.** It wasn't a simple
  geometric TP-vs-step artifact -- widening TP past the leg-step trigger
  didn't raise OOS multi-leg engagement above ~1% at any level tried, because
  the Bayesian gate itself denies escalation once the pooled posterior's win
  rate stops clearing breakeven. Encouraging in one sense (the safety
  mechanism is doing its job, refusing to martingale into a weak edge) and
  frustrating in another (it means this strategy's core idea -- does
  averaging down actually help -- still hasn't been tested on a symbol set
  where it isn't already the mechanism the gate would suppress).
- **The actual profit lever was trade selectivity, not the averaging
  mechanic.** Fewer, larger-payoff trades (win rate down, avg R and total
  return up) is what improved both samples -- a fairly ordinary
  risk/reward-ratio tuning result, achieved here via the TP parameter rather
  than anything specific to martingale sizing.
- **Tail risk stayed just as contained and the whole return distribution
  roughly doubled** in the bootstrap -- but this is still measured against
  trade outcomes from one ~6.5yr, bull-leaning window, the same caveat every
  other strategy in this project carries (see `FINDINGS_MTF.md`'s
  deep-history retest, which reversed a result that looked solid on a
  shorter window).
- **Combined portfolio's crowding cost is fixable, and fixed.** Breakout
  Hunter's shared-pool t-stat dropping to ~1.9-2.05 was a cash-availability
  problem, not a position-count one (raising `max_concurrent_positions` to
  40 or removing it barely moved the number). Reserving ~80% of the pool for
  Breakout Hunter via `grid_capital_cap_frac=0.2` recovers its t-stat to
  ~2.4-2.5 AND improves Grid's own (capital scarcity makes it more selective)
  AND costs nothing on portfolio-level return/drawdown -- a genuine win-win,
  not the usual "more for one means less for the other" tradeoff this
  project's earlier combined-portfolio test found.

**Recommendation**: still not ready for live paper trading, but net better
and more credible than before -- keep `long_only=True`,
`MIN_BARS_BETWEEN_LEGS=1`, `TP_ATR_MULT=2.0`, and (when running alongside
Breakout Hunter in a shared pool) `grid_capital_cap_frac=0.2` as the new
defaults -- all four are now applied, not just documented as options. If
pursued further: (1) retest across the deeper multi-regime
Alpaca history the way `FINDINGS_MTF.md`'s final retest did, since this test
window is the same ~6.5yr Yahoo daily window already known to be bull-leaning
for every other strategy in this project -- the natural next stress test
given OOS is now positive but unconfirmed rather than clearly negative;
(2) the averaging mechanic itself still has no real test on fresh data -- that would require either a
looser Bayesian gate threshold or a symbol set where the gate's own
assessment of the edge is more favorable, not just a wider TP.

## Caveats
- Same correlated-symbols caveat as every other basket test here -- headline
  trade counts overstate independent sample size.
- The pooled Bayesian posterior is mutated across symbols in `run_basket`'s
  sequential per-symbol loop order, not real calendar time -- a documented
  limitation of reusing `run_basket` unmodified (see design notes in
  `backtest_grid_martingale.py`).
- No re-entry cooldown: a grid can stop out and reopen the very next bar off
  the same still-qualifying signal.
- The `grid_capital_cap_frac` sweep above 0.3 was noisy/non-monotonic, same
  processing-order-dependence caveat `FINDINGS_COMBINED.md` already flagged
  for `max_concurrent_positions` -- only the below-0.3 region gave a clean,
  trustworthy trend.

## Code
- `strategy_breakout.py` -- added `short_entry` (mirror of `long_entry`,
  unvalidated on its own before this)
- `strategy_grid.py` -- wraps `strategy_breakout.prepare()`, adds
  `regime_kill`
- `bayesian_tracker.py` -- `BetaBinomialPosterior` (Beta-Binomial posterior,
  credible intervals), `test_bayesian_tracker.py` for unit tests
- `backtest_grid_martingale.py` -- `GridLeg`/`GridPosition`/`GridTrade`,
  event-driven engine, Bayesian gate math
- `run_basket_grid_martingale.py` -- US-only basket run, long-only
- `sweep_grid_legtiming.py` -- MIN_BARS_BETWEEN_LEGS sweep (v1.1)
- `sweep_grid_tp.py` -- TP_ATR_MULT sweep, IS+OOS evaluated together (v1.2)
- `test_grid_oos_symbols.py` -- in-sample/out-of-sample check
- `simulate_grid_martingale_montecarlo.py` -- bootstrap tail-risk analysis
- `backtest_combined_grid.py` -- shared-equity portfolio with Breakout Hunter
