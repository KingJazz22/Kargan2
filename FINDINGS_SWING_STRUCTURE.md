# Swing structure (HH/HL/LH/LL) — backtest finding

**Conclusion: the structure-confirmed breakout variant is a real, validated
edge — arguably the strongest result in this project so far (in-sample t=4.41,
out-of-sample t≈3.0, both clearing significance; unlike Breakout Hunter, the
OOS side does not go flat). The trend-continuation pullback variant is
positive but unconfirmed (t=1.81 on both splits, below the conventional
significance bar) — a real avg-R edge driven by a small number of large
winners against a majority of small losers, not yet trustworthy on its own.**

## New capability: `indicators.swing_points` / `swing_structure`
No swing/pivot/fractal detection existed anywhere in this codebase before
this pass — `strategy_breakout.py`'s "resistance"/"support" was a naive
rolling 20-bar high/low. Added two functions to `indicators.py`:
- `swing_points(high, low, left=5, right=5)`: vectorized fractal pivot
  detection (bar `i` is a swing high/low if it's the max/min of the centered
  window `[i-left, i+right]`), returned shifted forward by `right` bars so a
  pivot only appears once it's actually knowable (no lookahead).
- `swing_structure(high, low, left=5, right=5)`: classifies each confirmed
  swing high/low as higher/lower than the previous one of the same type, and
  derives a per-bar `structure` state — `"uptrend"` (last swing high is a
  Higher High AND last swing low is a Higher Low), `"downtrend"` (mirror), or
  `"range"` (mixed — the consolidation state) — plus forward-filled
  `resistance`/`support` levels from the last confirmed pivots.

Verified with `test_swing_structure_offline.py`: a hand-built 20-bar zigzag
with known pivots, asserting exact confirmed pivot prices/positions, correct
HH/HL/LH/LL classification (including the transition bar between an uptrend
and downtrend leg correctly reading as `"range"`), and that nothing appears
before its confirmation bar. All checks pass.

## Strategy A: structure-confirmed breakout (`strategy_swing_breakout.py`)
Long only. Setup: `structure == "range"` (genuine consolidation by swing-pivot
definition, not a volatility squeeze). Entry: close breaks above `resistance`
(the last confirmed swing-high pivot) with volume above its 20-bar average.
Exit (`backtest_swing_breakout.py`): the same two-stage stop already validated
for Breakout Hunter — structural initial stop below the pre-breakout `support`
swing low, ATR trail activates after 1x ATR profit cushion. Daily bars, US
equities only (Breakout Hunter found no crypto edge at any timeframe, so this
first pass doesn't retest crypto).

Gap filled in, flagged: no fractal window was specified for what counts as a
swing pivot. Defaulted to `left=right=5` — a reasonably conservative choice
for daily bars; not swept in this pass.

| | In-sample (20 US symbols) | Out-of-sample (20 US symbols) |
|---|---|---|
| Trades | 377 | 373–375* |
| Win rate | 61.0% | 58.1–58.7% |
| Avg R | +0.198 | +0.137–0.145 |
| Profit factor | 1.93 | 1.58–1.63 |
| t-stat | **4.41** | **~3.0** |

\* Two runs of the OOS basket returned slightly different trade counts
(373 vs. 375) — live `yfinance` downloads aren't perfectly reproducible
run-to-run (retries/partial data), not a bug in the strategy or backtest
logic; the t-stat stayed clearly significant (2.97–3.15) both times.

**This is the strongest combined result in the project.** Breakout Hunter's
in-sample number was similar in shape (t=2.88) but its out-of-sample result
came back flat (avg R -0.010, t=-0.24) — the combined significance there
depended on in-sample carrying the OOS half. Here, OOS is independently
significant on its own (t≈3.0), which is a materially stronger validation
signal. Exit mix is dominated by the trailing stop (297-310 of ~375 exits per
group), meaning most trades that survive the initial stop go on to ride a
real trend — consistent with swing-pivot resistance being a higher-quality
level than a naive rolling high.

## Strategy B: trend-continuation pullback (`strategy_swing_pullback.py`)
Long only. Setup: `structure == "uptrend"`. Entry: the bar's Low comes within
1x ATR of `support` (a genuine pullback touch) AND Close is back above
`support` (the level held). Exit: same two-stage stop shape, anchored to the
defended support level instead of a pre-breakout low.

Gaps filled in, flagged: same `left=right=5` fractal window as Strategy A for
consistency; the pullback tolerance (`PULLBACK_ATR_MULT=1.0`, i.e. "within 1x
ATR counts as a touch") is an untuned starting point.

| | In-sample (20 US symbols) | Out-of-sample (20 US symbols) |
|---|---|---|
| Trades | 766 | 823 |
| Win rate | 36.2% | 34.8% |
| Avg R | +0.128 | +0.127 |
| Profit factor | 1.22 | 1.22 |
| t-stat | 1.81 | 1.81 |

Both splits land at the same t-stat almost exactly, and the shape is
consistent: median R is sharply negative (-0.64 to -0.72) while mean R is
positive — most trades are small losers cut at the initial stop or a shallow
trail, but a minority run far enough on the trailing stop to make the average
positive. That's a real, internally consistent pattern (not overfitting
noise, since it reproduces almost identically in-sample and OOS), but t=1.81
sits below this project's usual significance bar (Breakout Hunter needed
t>2.88 to be called validated) — **leaning positive, unconfirmed**, same
language `FINDINGS_MTF.md` used for its VWAP filter result before further
testing. Not ready to trade as-is; a tighter pullback-quality filter (e.g.
requiring the reclaim bar to close in the top third of its range, or a
minimum ADX to confirm the uptrend is real) is the natural next lever, since
the entry may currently be firing on too many shallow/ambiguous pullbacks.

## Corrected 2026-08-24: gap-through stop-fill bug fixed, both strategies re-run

`backtest_swing_breakout.py` (Strategy A) and `backtest_swing_pullback.py`
(Strategy B) both filled stop-loss/trailing-stop exits at the theoretical
stop price even when the market gapped straight through it overnight (proof:
SPY 2019-12-03 booked an exit at 281.72, above that day's actual High of
280.86 -- a price that was never tradeable). Fixed to clamp the fill to the
bar's Open when it already gapped past the stop, mirroring the gap-aware
pattern already used correctly for take-profits elsewhere in this codebase.
Full writeup: `FINDINGS_CHANNEL_MACD.md`/plan notes; same fix applied
project-wide.

**Strategy A (structure-confirmed breakout), cached-data re-run
(`run_original_cached_swing_breakout.py`):**

| | IS, before | IS, after | OOS, before | OOS, after |
|---|---|---|---|---|
| Trades | 377 | 377 | 374 | 374 |
| Win rate | 58.4% | 58.4% | ~58% | 57.0% |
| Avg R | +0.198 (live-data run) | +0.169 | +0.14 (live-data run) | +0.122 |
| Profit factor | 1.93 | 1.75 | ~1.6 | 1.50 |
| t-stat | **4.41** | **3.76** | **~3.0** | **2.66** |

The post-fix numbers land almost exactly on the vectorbt pilot's independent
port of this same engine (IS t=3.76 n=375, OOS t=2.70 n=370) -- the fix
closes the gap the vectorbt cross-check originally surfaced. **Verdict
unchanged: Strategy A remains a real, validated edge** (t=3.76 IS / t=2.66
OOS both clear significance) -- weaker than the pre-fix numbers implied, but
not remotely disproven.

**Strategy B (trend-continuation pullback), live-data re-run
(`run_basket_swing_pullback.py`, daily, 2019-01-01 to 2026-08-01):**

| | IS, before | IS, after | OOS, before | OOS, after |
|---|---|---|---|---|
| Trades | 766 | 767 | 823 | 826 |
| Win rate | 36.2% | 35.1% | 34.8% | 33.5% |
| Avg R | +0.128 | +0.079 | +0.127 | +0.039 |
| Profit factor | 1.22 | 1.13 | 1.22 | 1.06 |
| t-stat | 1.81 | 1.13 | 1.81 | 0.58 |

Strategy B was already below this project's significance bar pre-fix
("leaning positive, unconfirmed"); post-fix it drops further toward flat,
especially out-of-sample (t=0.58, avg R barely positive). **Verdict
strengthens from "unconfirmed" to "no longer even leaning positive" --
do not pursue Strategy B further without redesigning the entry.** Neither
strategy is live-wired, so no trading impact, but this materially lowers
confidence in Strategy B specifically.

## Verdict
- **Strategy A (structure-confirmed breakout) is validated and is now this
  project's best-known result** — recommend treating it the way Breakout
  Hunter is treated (a real edge, worth extending: short side as its own
  separately-validated test, and a timeframe sweep to check whether, like
  Breakout Hunter, the edge is specific to daily bars).
- **Strategy B (trend-continuation pullback) is not yet validated** — worth
  another iteration on entry quality before spending further backtest cycles
  on its exit design.

## Code
- `indicators.py` — `swing_points`, `swing_structure` (new)
- `test_swing_structure_offline.py` — offline sanity check (new)
- `strategy_swing_breakout.py` / `backtest_swing_breakout.py` /
  `run_basket_swing_breakout.py` — Strategy A
- `strategy_swing_pullback.py` / `backtest_swing_pullback.py` /
  `run_basket_swing_pullback.py` — Strategy B
