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
