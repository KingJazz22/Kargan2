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

## Caveat
Same as every other basket test: the 20 US symbols and 8 crypto pairs are
correlated, so headline trade counts overstate independent sample size.

## Code
- `strategy_mtf.py` -- signal/exit rules, `long_only` param, lookahead-safe HTF
  alignment via shifted-index `merge_asof`
- `backtest_mtf.py` -- event-driven engine, takes `(ltf_df, htf_df)` tuple
- `data.py` -- adds `fetch_yfinance_weekly` (native interval, no lookback cap)
- `run_basket_mtf.py` -- 1H+4H basket run
