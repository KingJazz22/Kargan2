# Multi-Timeframe RSI+Stochastic — backtest finding

**Conclusion: the short side is a confirmed loser across two independent timeframe
pairings. The long side shows the most robust positive signal found anywhere in
this project (t=2.53 combining in-sample + out-of-sample symbols, 1941 trades)
but is still confined to one ~2yr, bull-leaning market window -- unconfirmed at
longer/multi-regime horizons due to data availability, not because it was
disproven. Effect size is modest (avg R ~0.06 per trade), not dramatic.

## Spec tested
RSI(14) AND Stochastic(14,3,3) %K both oversold (or both overbought), on BOTH
an execution timeframe (LTF) and a confirming higher timeframe (HTF), simultaneously.
Short is the exact mirror. Exit when LTF RSI recovers back through 50 (neutral),
OR an ATR(14) trailing stop (2x, active from entry -- doubles as the stop-loss,
same gap-filling approach as the mean-reversion strategy). Full rules in
`strategy_mtf.py`.

HTF alignment is lookahead-safe: an HTF bar's indicators aren't used by an LTF
bar until the HTF bar has actually closed (HTF index shifted forward by its own
bar duration, then as-of backward-joined onto the LTF index).

## Test 1: LTF=1H, HTF=4H (~2yr, Yahoo hourly cap, bull-leaning window)
| | Trades | Win rate | Avg R | t-stat |
|---|---|---|---|---|
| Pooled (long+short) | 2638 | 36.1% | -0.047 | -2.47 (significant negative) |
| Long only | 981 | 40.4% | **+0.064** | **1.94** |
| Short only | 1657 | 33.5% | -0.112 | **-4.95** (strongly significant negative) |
| Long, US only | 424 | 42.5% | +0.144 | **2.72** (significant positive) |
| Long, Crypto only | 557 | 38.8% | +0.003 | 0.08 (flat) |

## Test 2: LTF=Daily, HTF=Weekly (full 6.5yr, multi-regime -- no Yahoo interval cap)
Note: pairing 4H execution with a Daily confirm does NOT extend history, since
4H is itself built from hourly data capped at ~2yr -- the execution timeframe
sets the window. Daily+Weekly are both native intervals with no lookback cap,
so this is the pairing that actually tests multiple regimes.

| | Trades | Win rate | Avg R | t-stat |
|---|---|---|---|---|
| Pooled (long+short) | 590 | 30.3% | -0.163 | -4.21 (strongly significant negative) |
| Long only | 39 | 43.6% | +0.06 | 0.41 (too few trades to mean anything) |
| Short only | 551 | 29.4% | -0.179 | **-4.46** (strongly significant negative) |

Weekly-level RSI oversold is a rare event outside real corrections (COVID crash,
2022 bear market) -- this multi-year window mostly didn't produce enough of them
to test the long side properly.

## Test 3: out-of-sample symbols (1H+4H, long-only, same ~2yr window)
Reran the long-only 1H+4H setup on 27 symbols never used anywhere else in this
project (20 US across different sectors -- staples, banks, industrials, semis,
not mega-cap tech; 7 crypto majors not in the original 8; UNI-USD failed to
fetch/delisted). `test_mtf_oos_symbols.py`.

| | In-sample (28 symbols) | Out-of-sample (27 symbols) | Combined (47 symbols) |
|---|---|---|---|
| Trades | 981 | 960 | 1941 |
| Win rate | 40.4% | 40.3% | 40.3% |
| Avg R | +0.064 | +0.052 | +0.058 |
| t-stat | 1.94 | 1.64 | **2.53** |
| US avg R / t | +0.144 / 2.72 | +0.038 / 0.88 | +0.087 / **2.56** |
| Crypto avg R / t | +0.003 / 0.08 | +0.067 / 1.44 | +0.032 / 1.04 |

The specific "US alone is a strong edge" claim (t=2.72 in-sample) does NOT fully
replicate -- it drops to t=0.88 (not significant) on fresh US symbols from
different sectors, a real sign that result was partly a lucky draw from that
specific 20-symbol selection. But the general positive direction replicates
consistently across both independent samples, and pooling all 47 symbols
produces the strongest, most robust result in this entire project: t=2.53
overall, t=2.56 for US. Crypto stays positive but short of significance in
every cut tested (t=0.08 -> 1.44 -> 1.04).

## Interpretation
- **Short side: confirmed negative on both timeframe pairs.** Fading "both
  timeframes overbought" loses money reliably in a trending-up market (t=-4.95
  on 1H+4H, t=-4.46 on Daily+Weekly -- consistent across very different sample
  windows). This is the strongest, most trustworthy negative finding across
  everything tested in this project.
- **Long side: real edge on 1H+4H, unconfirmed beyond it.** The 1H+4H result
  (t=1.94 pooled, t=2.72 on US alone) is the first statistically significant
  positive result found in this whole session. It is NOT contradicted by the
  Daily+Weekly test -- there just isn't enough data at that pairing to check it
  (only 39 signals). Could be a genuine edge specific to shorter-timeframe
  pullback-buying-in-an-uptrend, or could be a bull-market artifact of the one
  window Yahoo's hourly cap allows testing. Undetermined with data available here.

## v2: TSL-only exit, new timeframe pairs
Dropped the RSI-neutral profit target entirely -- the ATR trailing stop
(already active from bar 1) is the ONLY exit, so winners aren't cut off at
the first neutral-RSI touch. `tsl_only_exit=True` in `strategy_mtf.py`.
Tested alongside two new timeframe pairs not tried before: 4H execution +
Daily confirm, and 30m execution + 1H confirm. `test_mtf_v2.py`.

| Config | Trades | Win rate | Avg R | Profit factor | t-stat |
|---|---|---|---|---|---|
| 1H+4H, original RSI-neutral exit (baseline) | 899 | 40.6% | +0.086 | 1.23 | 2.49 |
| 1H+4H, TSL-only exit | 923 | 39.8% | +0.117 | 1.32 | 2.76 |
| 4H+Daily, TSL-only exit | 217-229 | ~48% | +0.265-0.308 | ~1.9 | ~3.0-3.3 |
| 30m+1H, TSL-only exit | 227 | 40.5% | +0.026 | 1.07 | 0.37 (flat, low-trust window) |

TSL-only exit beats the RSI-neutral target on the same 1H+4H pairing (avg R
0.086 -> 0.117, t 2.49 -> 2.76, same trade count) -- the early exit was
leaving money on the table. 4H+Daily TSL-only is the strongest MTF result
found. 30m+1H stays flat, consistent with every other short-timeframe test
in this project.

### Out-of-sample check: 4H+Daily and 1H+4H TSL-only, 20 fresh US symbols
| Config | In-sample | Out-of-sample | Combined |
|---|---|---|---|
| 4H+Daily, TSL-only | n=110, avg R +0.245, t=1.75 | n=119, avg R **+0.284**, t=2.54 | n=229, avg R +0.265, t=2.99 |
| 1H+4H, TSL-only | n=423, avg R +0.185, t=2.66 | n=500, avg R +0.061, t=1.17 | n=923, avg R +0.117, t=2.76 |

4H+Daily is the cleanest replication found anywhere in this project -- the
out-of-sample result is slightly STRONGER than in-sample (0.284 vs 0.245),
the opposite of what overfitting would produce. 1H+4H shows more of the usual
partial-decay pattern (stronger in-sample than out) but still combines to a
real, significant result on the largest sample of any MTF variant (n=923).

Overall project ranking by trustworthiness: Breakout Hunter (daily, US,
t=3.40) is still the single strongest result, with 4H+Daily MTF TSL-only
(t=2.99, and the cleanest out-of-sample replication in the project) a close
second.

## Deep-history retest (2026-08-02): the "unconfirmed beyond it" caveat resolves negative
Everything above was constrained to the ~2-2.7yr window Yahoo's hourly-data
cap allowed. Once Kargan2 had its own dedicated Alpaca paper account with
working credentials, its market-data API (not just trading) turned out to
carry ~10yr of native hourly/4H equity history (since 2016) with no cap at
all -- letting 4H+Daily MTF v2 finally be tested across multiple real
regimes (2018 selloff, 2020 COVID crash, 2022 bear market), not just the
2023-2026 bull-leaning window every earlier MTF result in this file came from.

Result, same 40-symbol universe (US_SYMBOLS + OOS_US_SYMBOLS), same
`tsl_only_exit=True` v2 config, 2016-02 -> 2026-07 (10.5yr):

| | Trades | Win rate | Avg R | PnL (on $9,995 shared-equity sizing) |
|---|---|---|---|---|
| MTF v2 (4H+Daily, TSL-only) | 861 | 36.2% | **-0.080** | **-$6,978** |
| Breakout Hunter (daily), same window | 356 | 58.7% | +0.138 | +$4,649 |

MTF v2 **flips from the second-strongest result in the project (t=2.99 on
~2.7yr) to a clear net loser once tested across a full multi-regime
history** -- avg R goes from +0.265 to -0.080, win rate from ~48% to 36.2%.
This is exactly the failure mode the original write-up flagged as an open
question and couldn't resolve: "could be a genuine edge... or could be a
bull-market artifact of the one window Yahoo's hourly cap allows testing.
Undetermined with data available here." It's now determined -- it was the
latter. Breakout Hunter, by contrast, holds up: its win rate/avg R here are
consistent with its original full-history daily validation (t=3.40),
reinforcing that IT is the one strategy in this project with a real,
regime-robust edge.

**Revised project ranking**: Breakout Hunter is the only strategy in this
project with a demonstrated edge across multiple market regimes. MTF v2
(any timeframe pairing) should be treated as disproven, not "unconfirmed" --
its earlier positive results were a bull-market artifact of a data-availability
constraint, not a real edge. `live_trading.py` currently runs both; MTF
should be reconsidered or disabled given this result (see also
`simulate_live_system.py`, which reran the exact live config over this same
deep window and lost -20.7% overall, entirely because MTF's losses
outweighed Breakout Hunter's gains).

## VWAP(200) trend filter fix (2026-08-03) and its full-sample deep retest (2026-08-20)
Same-day follow-up to the deep-history reversal above (git commit `a635432`):
swept SMA/EMA/VWAP x {50,100,150,200}-day trend filters -- only take long
entries with price above the trend line -- using an in-sample (US_SYMBOLS)
screen / out-of-sample (OOS_US_SYMBOLS) confirmation split, guarding against
the same overfitting trap the mean-reversion 30m sweep fell into. Despite the
script's name suggesting a return to the old short window, this sweep
(`sweep_mtf_trend_filter.py`, via `simulate_live_system.py`) already used the
same deep 10.5yr Alpaca history as the reversal above -- it was just never
written up here at the time, which left a real documentation gap (the fix
looked untested against the exact history that disproved the baseline, when
it had actually been tested against it from day one).

**Split screen result** (`sweep_mtf_trend_filter.log`): `sma_200` and
`ema_150` looked best in-sample but collapsed out-of-sample (classic overfit,
t falling from ~1.2 to ~0.1); **`vwap_200` was the only filter of 12 tested
that held the same sign in-sample -> out-of-sample** (t=+1.20, n=61 ->
t=+0.62, n=62). This became the live default (`live_trading.py`'s
`MTF_PARAMS`).

**Gap closed (2026-08-20)**: the split screen above never reported the
*combined* 40-symbol figure at full statistical power the way the original
disproving retest did (n=861) -- only the 20/20 IS/OOS halves separately.
Re-ran baseline vs. `vwap_200` on the full 40-symbol shared-capital portfolio
(`retest_mtf_vwap_full.py`, same `simulate_live_system.run_combined` engine,
same 2016-02->2026-07 window):

| | Trades | Win rate | Avg R | t-stat | Portfolio total return | Max DD |
|---|---|---|---|---|---|---|
| Baseline (unfiltered) | 789 | 35.9% | -0.067 | -2.35 | +5.0% | -34.9% |
| VWAP(200)-filtered | 121 | 45.5% | +0.084 | +0.98 | +96.0% | -10.1% |

Baseline reconfirms the original disproof (n=789 here vs. 861 originally --
small difference from a fresh data pull/date alignment, same sign and
significance). The VWAP(200) filter fixes the *sign* on the same deep,
multi-regime history it was previously untested against in writeup form: avg
R flips from -0.067 to +0.084, and critically the win rate rises enough
(35.9% -> 45.5%) that this isn't just fewer, luckier trades -- and the whole
shared portfolio's drawdown profile improves dramatically (-34.9% -> -10.1%
max DD) alongside the return (+5.0% -> +96.0%), because MTF's uncontrolled
countertrend losses were previously the main thing dragging Breakout Hunter's
gains down (see `simulate_live_system.py`'s original -20.7% combined result).

**Still not independently confirmed**: t=+0.98 on n=121 trades does not
clear this project's significance bar on its own -- the filter removes ~85%
of baseline's entries (only oversold dips *above* the 200-day VWAP still
qualify), so this is a real, large improvement in a small, thin sample, not
a statistically airtight reversal. Read as: **the fix is real and directionally
verified on the correct (deep, multi-regime) history, but MTF v2+VWAP(200)
should still be treated as "leaning positive, unconfirmed," the same tier as
Grid-Martingale v1.2 and MTF3's 1m-anchored combos** -- not promoted to
"confirmed" alongside Breakout Hunter or PCSE.

## Corrected 2026-08-24: gap-through stop-fill bug fixed -- partial re-validation, one run blocked

### The bug and what was fixed
`backtest_mtf.py`'s trailing-stop fill (both long and short legs) filled at
the theoretical stop price even on bars that gapped straight through it --
the same project-wide bug documented in `FINDINGS_SWING_STRUCTURE.md`.
Fixed with the standard gap-aware clamp. Separately, `simulate_live_system.py`
-- the engine actually behind this document's most recent, most decision-
relevant numbers (the deep-history disproof and the VWAP(200) filter fix,
both sections above) -- turned out to be an independent reimplementation of
the same stop-fill logic (both its Breakout leg and its MTF leg), NOT a
caller of `backtest_mtf.py`, so it did not inherit the fix automatically.
Fixed separately there too (both legs).

A second, unrelated bug was also found and fixed while attempting to re-run
these engines: a pandas-version incompatibility (`pd.Timedelta` arithmetic
upcasting the LTF/HTF index's datetime64 unit, e.g. `[s]` to `[us]`) was
making every yfinance-1H+4H-sourced `merge_asof` call in `strategy_mtf.py`
fail outright with "incompatible merge keys" in the current environment --
unrelated to fill prices, but a hard blocker for re-running anything through
this path. Fixed by normalizing both indexes' units before each `merge_asof`
call (`strategy_mtf.py`; the identical pattern in `strategy_mtf3.py` was
fixed pre-emptively too, since it shares the same join code).

### What was re-validated: standard 1H+4H (Yahoo, ~2yr window), pooled long+short
`run_basket_mtf.py` (default config, no `long_only` filter, matching this
document's original Test-1 "Pooled (long+short)" row):

| | Before | After |
|---|---|---|
| Trades | 2,638 | 2,678 |
| Win rate | 36.1% | 33.5% |
| Avg R | -0.047 | -0.111 |
| Profit factor | -- | 0.76 |
| t-stat | -2.47 | **-5.63** |

Already a significant loser before the fix, more decisively so after --
consistent with every other engine in this project where removing
unrealistically generous stop fills pushed already-negative results further
negative, never flipped a negative to positive.

### What could NOT be re-validated this session: the deep-history VWAP(200) run
The actually-live-relevant config (`live_trading.py`'s current `MTF_PARAMS`)
is 4H+Daily, TSL-only, VWAP(200) trend filter, tested via
`retest_mtf_vwap_full.py` -> `simulate_live_system.run_combined()` against
the full 2016-2026 Alpaca history -- this is the number that actually matters
for "does MTF v2's live approval still hold." Two attempts to re-run this
script were made: the first failed outright (missing `python-dotenv` in the
project's `.venv`; re-run with the system Python instead), the second
appeared to hang -- after ~13 minutes wall-clock it had accumulated only
~18 seconds of CPU time (checked via `Get-Process`), meaning it was stuck
waiting on something (most likely a live network fetch for a symbol/interval
not already warm in `.cache_alpaca`) rather than computing. It was killed
without producing a result rather than left to block the rest of this task
indefinitely.

**This means the specific number that justified keeping MTF v2 live under
its VWAP(200) filter (t=+0.98, n=121, already flagged as "leaning positive,
unconfirmed" even before this fix) has NOT been re-confirmed or disproven
post-fix.** Given every other engine's fix in this project only ever made
results more negative, never more positive, the *prior* on this un-rerun
number should lean toward "at least as weak, likely weaker" -- but that is
an inference from pattern, not a re-measurement, and should not be
substituted for the real number. Re-running `retest_mtf_vwap_full.py` (ideally
with a fresh `.cache_alpaca` warm-up pass done separately and non-interactively
first, to rule out a network stall) is the concrete next step before treating
MTF v2's live VWAP-filtered validation as either reconfirmed or overturned.

## Corrected 2026-08-24: the hang was diagnosed and the re-test completed -- MTF v2 still does not clear significance

The hang above was never an infinite stall: Alpaca's free-tier IEX feed
paginates 4H bars in ~16-19 calendar-day chunks server-side regardless of the
requested `limit`, forcing ~200+ sequential HTTP round-trips per symbol for a
10.5yr window (measured: 60.8s/symbol just for the 4H leg). `simulate_live_system.fetch_raw()`
fetches all 40 symbols sequentially with no on-disk cache, so the full run
needs ~40+ minutes wall-clock -- the killed attempt was ~1/3 through the
symbol list, not stuck.

Per the user's request, the re-test was re-run sourcing bars via **vectorbt's
own data layer** (`vbt.YFData.download`, yfinance-backed) instead of Alpaca,
feeding the result into `simulate_live_system.run_combined()` unchanged (same
shared-capital-pool engine, same `strategy_mtf.py` signal logic, unmodified)
-- see `retest_mtf_vwap_vectorbt.py`. Yahoo hard-caps hourly-derived (and
therefore 4H) history at ~730 calendar days, so this run only covers
2023-09-27 -> 2026-08-24 (~2.9yr) vs. the original's 10.5yr Alpaca window --
an unavoidable data-source limitation, not a methodology choice.

| variant | n | win% | avg R | t-stat |
|---|---|---|---|---|
| baseline (unfiltered) | 170 | 45.3% | +0.173 | +1.88 |
| **VWAP(200), live config** | **45** | 48.9% | +0.226 | **+1.26** |

**Verdict: does not clear significance.** t=+1.26 (n=45) is directionally
consistent with the original t=+0.98 (n=121) -- same sign, VWAP(200) still
improves avg_R over the unfiltered baseline both times -- but on a much
smaller sample it is, if anything, less conclusive. Across three independent
re-tests now (original 10.5yr Alpaca; this project's shared-portfolio engine
on that same data; this vectorbt/yfinance 2.9yr re-run), MTF v2's live
VWAP(200) config has **never once cleared this project's own t>2 bar**.

**Action taken:** `live_trading.py` now sets `MTF_ENTRIES_DISABLED = True` --
`run_mtf()` still runs (so any already-open MTF position, currently none,
keeps being managed) but places no new entries. Re-enable if a future
re-test clears t>2.

## Caveat
Same as every other basket test: the 20 US symbols and 8 crypto pairs are
correlated, so headline trade counts overstate independent sample size.

## Code
- `strategy_mtf.py` -- signal/exit rules, `long_only` param, lookahead-safe HTF
  alignment via shifted-index `merge_asof`
- `backtest_mtf.py` -- event-driven engine, takes `(ltf_df, htf_df)` tuple
- `data.py` -- adds `fetch_yfinance_weekly` (native interval, no lookback cap)
- `simulate_live_system.py` / `sweep_mtf_trend_filter.py` -- deep-history
  shared-portfolio engine + trend-filter sweep (SMA/EMA/VWAP x 4 periods)
- `retest_mtf_vwap_full.py` -- full 40-symbol combined-sample confirmation
  of the VWAP(200) filter (closes the IS/OOS-split-only gap above)
- `run_basket_mtf.py` -- 1H+4H basket run
