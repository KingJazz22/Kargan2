# 3-Timeframe MTF Alignment — backtest finding

**Conclusion: the deciding variable isn't "2 vs 3 timeframes" — it's whether 1-minute
bars are the execution timeframe. Every combo (2-TF or 3-TF) anchored on a 1m LTF
shows a positive edge; every combo anchored on anything coarser (5m/10m/15m/30m) is
a loser, often a strongly significant one (t down to -9). Adding a 3rd confirming
timeframe on top of a 1m-anchored pair does NOT increase trade frequency — it can't,
it's a stricter AND — but it does act as a real quality filter: fewer trades, higher
avg R, and the effect held up (mostly got stronger) on a held-out symbol set. "More
frequent AND more profitable" is not jointly achievable by adding a 3rd timeframe:
you get one dial, not two. This entire test is scoped to a 2-year window by explicit
choice (see Caveats) — the project's track record is that short-window MTF results
have previously flipped sign on deep multi-regime history, so treat this as a
preliminary signal, not a confirmed edge.**

## Spec tested
RSI(14) AND Stochastic(14,3,3) %K both oversold (or both overbought), simultaneously,
on THREE timeframes at once (execution/LTF + two confirming higher frames, MTF+HTF) —
`strategy_mtf3.py`, extending the existing 2-TF `strategy_mtf.py` alignment logic to a
3-way AND. Long-only, TSL-only exit (2x ATR trailing stop, active from bar 1, no
RSI-neutral profit target) — the "v2" config `FINDINGS_MTF.md` found strictly beats
the RSI-neutral exit on every 2-TF pairing tested. Same fixed-fractional position
sizing (1% risk per trade off the ATR stop distance) and next-bar-open fill
convention as every other strategy in this repo — `backtest_mtf3.py` is a line-for-line
copy of `backtest_mtf.py`'s event loop, just unpacking a 3-tuple instead of a 2-tuple.

Universe: 76 symbols (60 stocks/index ETFs + 16 crypto pairs), combining this
project's existing validated lists with 20 fresh staples/healthcare/utilities/telecom
names from `screen_stable_stocks.py` for diversity — `mtf3_universe.py`. Window:
2024-08-01 to 2026-08-01 (2 years), all 7 timeframes {1m,5m,10m,15m,30m,1h,4h}
fetched natively from Alpaca (no resampling needed) and cached offline via
`prefetch_mtf3_data.py` — every sweep below reads from local disk, no live API calls.

## Phase 1: all C(7,3)=35 timeframe triples, full 76-symbol universe
Ranked by t-stat vs zero. Full table in `mtf3_sweep_phase1.csv`; top and bottom shown:

| Combo | Trades | Win rate | Avg R | Profit factor | t-stat |
|---|---|---|---|---|---|
| 1m+5m+1h | 9,163 | 38.2% | **+0.014** | 1.11 | **2.77** |
| 1m+5m+10m | 42,101 | 38.2% | +0.005 | 1.04 | 2.39 |
| 1m+10m+1h | 9,881 | 37.8% | +0.011 | 1.09 | 2.31 |
| 1m+30m+4h | 4,278 | 38.0% | **+0.021** | **1.14** | 2.22 |
| 1m+1h+4h | 5,922 | 37.4% | +0.018 | 1.13 | 2.22 |
| 1m+15m+4h | 3,731 | 37.7% | +0.008 | 1.04 | 0.81 |
| ... | | | | | |
| 15m+1h+4h | 2,154 | 39.3% | -0.017 | 0.94 | -0.97 |
| 5m+10m+1h | 7,462 | 39.0% | -0.011 | 0.96 | -1.43 |
| 30m+1h+4h | 2,023 | 38.5% | -0.057 | 0.84 | -3.18 |
| 15m+30m+1h | 7,034 | 39.4% | -0.040 | 0.87 | -4.46 |
| 5m+10m+15m | 24,157 | 38.1% | -0.033 | 0.86 | **-8.49** |
| 10m+15m+30m | 12,487 | 38.9% | -0.046 | 0.84 | -7.22 |

**Every one of the 15 combos with LTF=1m is net positive (avg R +0.005 to +0.021).
Every one of the 20 combos with LTF ≥ 5m is net negative** (avg R -0.007 to -0.057,
several with t < -4). This is a clean, total split — not a marginal skew.

Within the 1m-anchored combos, there's a clear frequency/quality tradeoff:
tightly-spaced triples (1m+5m+10m, 1m+5m+15m, ...) fire 3-4x more trades but with
avg R shrunk to ~0.005 — barely above the level realistic commissions/slippage
would erase (not modeled here, see Caveats). Widely-spaced triples (1m+30m+4h,
1m+1h+4h) trade far less often but carry 2-4x the avg R per trade.

## Phase 2: out-of-sample confirmation, top 5 combos
Symbol-based IS/OOS split (window is fixed at 2yr, so a time split isn't
meaningful): in-sample = 48 symbols already used elsewhere in this project,
out-of-sample = 28 fresh symbols never used anywhere else in the codebase.

| Combo | IS trades | IS avg R | IS t | OOS trades | OOS avg R | OOS t |
|---|---|---|---|---|---|---|
| 1m+5m+1h | 6,671 | +0.011 | 1.95 | 2,492 | +0.023 | 2.02 |
| 1m+5m+10m | 28,839 | +0.001 | 0.30 | 13,262 | +0.014 | 3.42 |
| 1m+10m+1h | 7,228 | +0.004 | 0.69 | 2,653 | +0.033 | 2.72 |
| 1m+30m+4h | 3,110 | +0.002 | 0.20 | 1,168 | **+0.072** | 2.94 |
| 1m+1h+4h | 4,246 | -0.003 | -0.32 | 1,676 | **+0.070** | 3.40 |

All 5 combos are positive out-of-sample, 4 of 5 get *stronger* OOS than IS (the
opposite of what overfitting would produce — same "cleanest replication" pattern
`FINDINGS_MTF.md` noted for 4H+Daily). `1m+5m+1h` is the most balanced candidate:
positive and significant on both sides (IS t=1.95, OOS t=2.02) with the highest
combined trade count of the top tier that isn't diluted to near-zero avg R.

## Baseline: existing 2-TF MTF, same 76-symbol universe, same 2yr window
Re-ran the *unmodified* `backtest_mtf.py`/`strategy_mtf.py` (not the old
`FINDINGS_MTF.md` numbers, which used a different universe/window/data source) across
all C(7,2)=21 pairs from the same 7 timeframes, for an apples-to-apples comparison.

| Combo | Trades | Win rate | Avg R | Profit factor | t-stat |
|---|---|---|---|---|---|
| 1m+5m | 79,028 | 38.5% | +0.004 | 1.04 | **2.86** |
| 1m+1h | 18,896 | 37.6% | +0.008 | 1.08 | 2.44 |
| 1m+10m | 48,298 | 37.9% | +0.003 | 1.03 | 1.74 |
| 1m+15m | 36,350 | 38.0% | +0.004 | 1.04 | 1.69 |
| 1m+30m | 24,877 | 37.9% | +0.004 | 1.04 | 1.44 |
| 1m+4h | 15,545 | 36.4% | -0.001 | 0.99 | -0.22 |
| 1h+4h | 2,213 | 36.3% | -0.087 | 0.77 | -4.87 |
| 30m+1h | 6,836 | 37.2% | -0.089 | 0.75 | **-9.35** |

Same pattern as the 3-TF sweep: **only pairs with LTF=1m are net positive**; every
2-TF pair without a 1m leg is a loser (down to t=-9.35 for 30m+1h). The pairing
closest in spirit to the currently-live `1h+4h`/`4H+Daily`-style MTF config (i.e. no
fine execution leg) is a clear loser here too — consistent with the deep-history
reversal `FINDINGS_MTF.md` already found for that style of pairing.

## 3-TF vs 2-TF, head to head
Isolating the components of the best-balanced 3-TF candidate:

| Config | Trades | Avg R | t-stat |
|---|---|---|---|
| 2-TF: 1m+5m | 79,028 | +0.004 | 2.86 |
| 2-TF: 1m+1h | 18,896 | +0.008 | 2.44 |
| **3-TF: 1m+5m+1h** | **9,163** | **+0.014** | **2.77** |
| 3-TF: 1m+30m+4h (best avg R) | 4,278 | +0.021 | 2.22 |

Adding the 3rd alignment condition roughly halves-to-thirds the trade count of the
best comparable 2-TF pair, but nearly doubles the avg R per trade (0.008 → 0.014,
or up to 0.021 for the highest-quality triple). **This directly contradicts the "more
frequent AND more profitable" goal as a joint outcome** — the 3rd timeframe is a
pure quality filter, not a frequency booster. If raw trade count is the priority,
the plain 2-TF 1m+5m pair wins (79k trades); if per-trade edge is the priority, the
3-TF triples win, and comfortably beat every 2-TF pair on avg R.

## Interpretation
- The real finding isn't "3 timeframes beat 2" — it's that **a 1-minute execution
  leg is necessary** for any edge to show up at all in this window/universe, on
  either the 2-TF or 3-TF version of this strategy. Every combo without it lost
  money, several very significantly.
- Given a 1m execution leg, a 3rd confirming timeframe is a legitimate, OOS-replicating
  quality filter: fewer but better trades. `1m+5m+1h` (n=9,163, avg R +0.014, t=2.77
  pooled; IS t=1.95, OOS t=2.02) is the best-balanced candidate; `1m+30m+4h` and
  `1m+1h+4h` have the highest avg R (+0.021, +0.018) if a lower trade count is
  acceptable.
- The strategy does not deliver on "more frequent" relative to a comparable-quality
  2-TF alternative — it delivers "more profitable per trade, less frequent." Whether
  that's actually what's wanted depends on which the user prioritizes; the two goals
  don't move together here.

## Corrected 2026-08-24: gap-through stop-fill bug fixed -- top candidates flip negative

`backtest_mtf3.py`'s trailing-stop fill (both long and short) filled at the
theoretical stop price even on bars that gapped straight through it, the same
project-wide bug documented in `FINDINGS_SWING_STRUCTURE.md`. Given this
strategy's shortest-timeframe execution leg is 1-minute bars, gap-through
events are far more frequent here than in any daily-bar strategy in this
project, so the correction hits harder. Re-ran the top 3 Phase-2 candidates
(`run_basket_mtf3_confirm.py 3`, same IS/OOS symbol split):

| Combo | IS avg R, before | IS avg R, after | IS t, before | IS t, after | OOS avg R, before | OOS avg R, after | OOS t, before | OOS t, after |
|---|---|---|---|---|---|---|---|---|
| 1m+5m+1h | +0.011 | **-0.007** | 1.95 | **-1.39** | +0.023 | **-0.008** | 2.02 | **-0.84** |
| 1m+5m+10m | +0.001 | **-0.014** | 0.30 | **-6.14** | +0.014 | **-0.009** | 3.42 | **-2.66** |
| 1m+10m+1h | +0.004 | **-0.014** | 0.69 | **-2.79** | +0.033 | **-0.001** | 2.72 | **-0.13** |

**All three top candidates flip sign, in-sample AND out-of-sample.** This is
the single largest correction found anywhere in this project's gap-fill fix
pass -- the previous "leaning positive, best-balanced candidate" read on
`1m+5m+1h` (t=1.95 IS / 2.02 OOS) was substantially an artifact of unrealistic
stop fills, not a real edge. `1m+5m+10m` in particular is now a large-sample,
strongly significant LOSER (n=28,839 IS, t=-6.14) rather than a flat result.

**Revised verdict: none of this file's 3-TF candidates hold up. Every combo
tested lost money on a fixed-realism basis, and the "1m execution leg is
necessary for any edge" interpretation should itself be treated as
disproven, not just unconfirmed** -- the 2-TF baseline table above (all
1m-anchored pairs) was generated by the same buggy engine and has not been
re-run, so it should also be read with this correction in mind pending a
fuller re-validation. This strategy was never live-wired, so there is no
trading impact, but the earlier "best-balanced candidate" framing should not
be carried forward into any future work on this file.

## Caveats
- **2-year window only**, by explicit choice for this test. `FINDINGS_MTF.md`
  documents the standing cautionary precedent in this exact project: the 2-TF MTF
  strategy's best-looking short-window result (4H+Daily, TSL-only, t=2.99 on
  ~2.7yr) flipped to a clear net loser (avg R -0.080) once retested on 10.5 years of
  multi-regime Alpaca history. Nothing here has been checked against that longer
  horizon — these results should be read as preliminary, not confirmed.
- **No transaction costs or slippage modeled.** The thinnest-edge combos (1m+5m+10m
  and similar, avg R ~0.005/trade across tens of thousands of trades) are exactly
  the kind of result realistic minute-bar commissions/slippage could plausibly wipe
  out; the wider-spaced, higher-avg-R triples (1m+30m+4h, 1m+1h+4h) are more robust
  to this risk simply by trading far less often.
- Same pooling caveat as every other `FINDINGS_*.md` in this repo: 76 symbols
  (many correlated, e.g. mega-cap tech, crypto majors) are not 76 independent
  samples, so headline trade counts overstate real statistical power.
- 28 of the 532 requested symbol/timeframe fetches failed (mostly a handful of
  crypto pairs not available on Alpaca, e.g. ATOM/USD, ETC/USD, XLM/USD) —
  `basket.run_basket` skips these gracefully; they're a small fraction of the
  76-symbol universe and don't materially affect the pooled results.
- This is a backtest/research deliverable only — nothing here has been wired into
  `live_trading.py`.

## Corrected 2026-08-26: full 56-combo x 7-threshold grid re-run on a validated vectorbt port — every single result confirms the reversal, zero survivors

The 2026-08-24 correction above only spot-checked the top 3 Phase-2 candidates.
The full Phase 1 grids (`mtf3_sweep_phase1.csv`'s 35 3-TF combos, plus the 2-TF
baseline table, plus every RSI- and Stochastic-threshold variant in
`FINDINGS_RSI_SWEEP.md`/`FINDINGS_STOCH_SWEEP.md`) were never re-run on the
fixed engine until now. This correction re-runs **all of it**: 56 timeframe
combos (35 3-TF + 21 2-TF) x 7 threshold settings (standard 30/70+20/80, three
tighter RSI variants, three tighter Stochastic variants) x the same 72-of-76
symbol universe (4 crypto symbols — BNB/USD, ATOM/USD, ETC/USD, XLM/USD — are
consistently absent from the offline cache across every timeframe and are
skipped, same as `run_basket`'s per-symbol try/except would do) = 392
combo/threshold cells, via a new vectorbt 1.1.0 port (`mtf3_vbt_engine.py`)
of `backtest_mtf3.py`/`backtest_mtf.py`'s execution engine, since re-running
the original Python-loop engine over the full grid would take on the order of
a day (see Wall-clock below).

**Headline: 0 of 392 combo/threshold cells clear t>2 anywhere in the grid.**
Every combo that was reported positive at standard thresholds is gone:

| tf_type | combos positive pre-fix (standard thresholds) | still positive post-fix |
|---|---|---|
| 3-TF | 15 (all 1m-anchored) | 2, both statistically insignificant (t=0.51, t=0.12) |
| 2-TF | 5 (all 1m-anchored) | 0 |

Old vs. new, 3-TF standard-threshold ranking (full table, was `mtf3_sweep_phase1.csv`):

| Combo | Old n | Old avg R | Old t | New n | New avg R | New t |
|---|---|---|---|---|---|---|
| 1m+5m+1h | 9,163 | +0.014 | 2.77 | 8,979 | -0.000 | -0.09 |
| 1m+5m+10m | 42,101 | +0.005 | 2.39 | 41,229 | -0.004 | -2.17 |
| 1m+10m+1h | 9,881 | +0.011 | 2.31 | 9,661 | -0.001 | -0.24 |
| 1m+1h+4h | 5,922 | +0.018 | 2.22 | 5,782 | -0.009 | -1.23 |
| 1m+30m+4h | 4,278 | +0.021 | 2.22 | 4,182 | +0.004 | 0.51 |
| 1m+15m+1h | 10,916 | +0.009 | 2.03 | 10,690 | +0.001 | 0.12 |
| ...remaining 9 previously-positive 3-TF combos... | | | | all | negative | t as low as -2.11 |
| 5m+10m+15m (worst, unchanged direction) | 24,157 | -0.033 | -8.49 | 23,678 | -0.039 | -9.58 |

Old vs. new, 2-TF standard-threshold ranking (full table, was `mtf2_baseline_same_window.csv`):

| Combo | Old n | Old avg R | Old t | New n | New avg R | New t |
|---|---|---|---|---|---|---|
| 1m+5m | 79,028 | +0.004 | 2.86 | 77,417 | -0.005 | -3.50 |
| 1m+1h | 18,896 | +0.008 | 2.44 | 18,464 | -0.004 | -1.16 |
| 1m+10m | 48,298 | +0.003 | 1.74 | 47,278 | -0.006 | -2.95 |
| 1m+15m | 36,350 | +0.004 | 1.69 | 35,614 | -0.004 | -2.00 |
| 1m+30m | 24,877 | +0.004 | 1.44 | 24,361 | -0.004 | -1.37 |
| 30m+1h (worst, unchanged direction) | 6,836 | -0.089 | -9.35 | 6,711 | -0.121 | -12.24 |

**The "1m execution leg matters" pattern survives directionally but not as an
edge**: 1m-anchored combos are still meaningfully less negative than non-1m
combos at every threshold tested (e.g. standard: pooled avg R -0.0046 vs.
-0.0448, trade-count-weighted), so the *relative* ordering this file's original
Phase 1 found is real. But the *absolute* claim — "every 1m-anchored combo is a
net winner" — is false on the corrected engine. 1m-anchored combos are now a
small loser too, just a smaller one than everything else. There is no longer
any threshold/combo/timeframe-type cell in this entire 56x7 grid that
qualifies as a discovered edge.

See `FINDINGS_RSI_SWEEP.md` and `FINDINGS_STOCH_SWEEP.md` for their own
correction sections (including the Stoch 5/95, 1m+5m headline — this
investigation's single strongest number, t=4.72 — which does **not** survive
either: new t=-0.26).

**Engine/validation notes** (see `FINDINGS_STOCH_SWEEP.md`'s correction section
for the full validation writeup): the vectorbt port was validated against
`backtest_mtf3.py` on the standard 1m+5m+1h config before being trusted for
the full grid. Tier 1 (per-trade, SPY/AAPL/NVDA) matched 94.4% (target ≥95%);
Tier 2 (pooled stats, full 72-symbol universe) passed on win_rate/avg_R/profit_factor
but missed strict tolerance on num_trades/t-stat. Every mismatch traces to one
understood, irreducible structural gap: vectorbt's `from_signals` pipeline
cannot check a stop-loss fill on the same bar a position enters (the stop
check for a bar runs before that bar's entry order is processed), so the
small fraction of trades that would have reversed and hit their stop within
their own entry bar are missed, along with the re-entries that would have
followed. This makes the port's numbers systematically *less negative* than
the true original-engine numbers (the missed trades are disproportionately
fast, adverse ones) — meaning every number in this correction is, if
anything, a mildly conservative (optimistic) read of the reversal, not an
inflated one. The direction and magnitude of the reversal is not in question.

**Wall-clock**: the full corrected grid (54 new combo cells beyond the 2 used
for a pre-launch smoke test, x7 thresholds x 72 symbols, indicators cached
once per symbol/timeframe and reused across every combo that shares a
timeframe) ran in 2,778s (~46 minutes) end to end. Two direct head-to-head
timings against the original engine on the same data: the heaviest combo
(1m+5m+1h) took 120.9s per threshold in the original engine (7 thresholds ≈
846s ≈ 14.1 min for that one combo alone) vs. 92.2s for all 7 thresholds
bundled in one vectorbt call (~9.2x faster); the lightest combo tested
(30m+1h+4h) took 9.9s/threshold originally (7 ≈ 69s) vs. 15.6s bundled
(~4.4x faster, since vectorbt's fixed per-call overhead matters more on
small data). A full from-scratch re-run of the original sequential sweep
across all 56 combos x 7 thresholds was not attempted directly (extrapolated
from the above, it would run on the order of several hours to a day, vs. 46
minutes for the vectorbt grid) — not worth burning the wall-clock just to
confirm a ratio already bounded by two direct measurements at the light and
heavy ends of the combo-weight range.

## Code
- `strategy_mtf3.py` — 3-way RSI+Stochastic alignment signal, extends
  `strategy_mtf.py`'s lookahead-safe shifted-`merge_asof` pattern to a 2nd confirming
  timeframe
- `backtest_mtf3.py` — event-driven engine, `(ltf_df, mtf_df, htf_df)` tuple contract
  (also: switched both this and `backtest_mtf.py`'s event loop from `iterrows()` to
  `itertuples()` — ~11x faster, same trade-for-trade output, needed once LTF bar
  counts hit the 100k-400k range for 1m data)
- `mtf3_universe.py` — 76-symbol universe + IS/OOS split
- `mtf3_data.py` — shared timeframe fetcher/cache-key layer
- `prefetch_mtf3_data.py` — offline cache population (76 symbols x 7 timeframes)
- `sweep_mtf3_combos.py` — Phase 1, all 35 combos (`mtf3_sweep_phase1.csv`)
- `run_basket_mtf3_confirm.py` — Phase 2, top-5 IS/OOS confirmation
  (`mtf3_sweep_phase2_confirm.csv`)
- `run_basket_mtf2_baseline.py` — 2-TF baseline, same universe/window
  (`mtf2_baseline_same_window.csv`)
- **New, 2026-08-26**: `vbt_synthetic_probe_mtf3.py` — synthetic-data validation
  of the vectorbt trailing-stop port (long AND short, 5/5 cases matched);
  `mtf3_vbt_engine.py` — the vectorbt port itself; `validate_vectorbt_mtf3.py` —
  Tier 1/Tier 2 validation against `backtest_mtf3.py`; `run_full_grid_mtf3_vbt.py` —
  the full 56-combo x 7-threshold grid runner (`mtf3_vbt_full_grid_CORRECTED.csv`);
  `compare_corrected_grid.py` — old-vs-new comparison/reporting
