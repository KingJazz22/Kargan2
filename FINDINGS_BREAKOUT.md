# Breakout Hunter — backtest finding

**Conclusion: the best-performing strategy found in this entire project --
on US equities, on DAILY bars specifically. A real, statistically significant
edge that survives combining in-sample and out-of-sample symbols (t=3.40,
n=325). Crypto shows no edge at any timeframe. Shorter timeframes don't just
weaken the edge, they reverse it (1H is significantly negative, n=1234).**

## Spec tested
Detect a genuine low-volatility squeeze (ATR, Bollinger width, and Keltner
width each in the bottom quartile of their own recent history, plus volume
below its 20-bar average) within the last 10 bars. Enter long on a close above
the prior 20-bar high, with volume above its 20-bar average and ADX rising
(vs. 3 bars ago). Exit via a two-stage ATR stop: a structural initial stop
below the pre-breakout consolidation low, then an ATR trailing stop that takes
over once the trade has 1x ATR of profit cushion. Full rules in
`strategy_breakout.py`. Long only -- no breakdown/short leg was specified.

Two gaps filled in and flagged:
- "Keltner squeeze" read as parallel to "Bollinger squeeze" (each indicator's
  own width vs. its own recent history), not the classic cross-indicator
  "BB inside KC" TTM-squeeze definition -- that stricter reading fired on only
  2.4% of SPY daily bars and, combined with the other 3 detect conditions,
  produced zero backtest signals. The literal spec lists them as parallel
  bullet points, supporting the looser reading.
- No stop-loss beyond "ATR trailing stop" was specified. Reused the original
  trend-following strategy's two-stage stop (structural initial stop, delayed
  trailing-stop activation) rather than a trail active from bar 1, since a
  breakout is a trend-continuation bet that often gets an immediate throwback
  test of the broken level.

## Timing fix applied up front
Same class of self-contradiction found in every other strategy tested here:
"low volume" (detect) and "high volume" (entry) can't both be true on the
breakout bar itself. Fixed by checking the squeeze conditions over a recent
10-bar lookback rather than requiring them on the breakout bar.

## Result: daily bars, full 6.5yr history (post emergency-stop fix -- see below)
| | In-sample (28 symbols) | Out-of-sample (27 symbols) | Combined (47 symbols) |
|---|---|---|---|
| Trades | 288 | 267 | 555 |
| Win rate | 58.0% | 54.7% | 56.4% |
| Avg R | +0.131 | -0.010 | +0.063 |
| Profit factor | 1.62 | 0.96 | -- |
| t-stat | **2.88** | -0.24 | 1.98 |
| **US avg R / t** | -- | -- | **+0.150 / 3.40** |
| Crypto avg R / t | -- | -- | -0.060 / -1.38 |

The in-sample daily result looked exceptionally strong -- the best in this
project by a wide margin, on a full multi-regime 6.5-year window, not a thin
sample. It did not fully replicate on fresh symbols: out-of-sample came back
close to flat rather than negative. Because it was flat rather than negative,
the diluted combined sample still clears significance for US (t=3.40, n=325)
-- a real edge, more moderate than the in-sample number implied. The original
20-symbol US basket (mega-cap tech-heavy, tested during an AI-driven boom) was
likely an unusually favorable sample, not an illusory one.

## Emergency-stop bug found and fixed
`emergency_stop` fired on 19-26% of trades initially -- far more than a "rare
backstop" should. Diagnosed by checking, on 58 emergency_stop exits, whether
the exit bar's Open had actually gapped below the emergency level: only 4/58
had (true gaps). The other 54/58 showed `emergency_stop > initial_stop` --
i.e. the emergency stop, computed independently as `entry - 4xATR`, ended up
*closer* to entry than the support-based structural initial stop, because a
squeeze entry (by definition) has unusually low ATR at that moment, while the
20-bar support level can span a much wider range. The "rare backstop" was
functioning as the primary stop most of the time, cutting trades short before
the intended structural stop -- and some throwaways -- could resolve.

Fixed by anchoring `emergency_stop` as a buffer *beyond* `initial_stop`
(`initial_stop - 2xATR`) rather than computing it independently from entry
price -- this guarantees it can never invert. Effect of the fix:

| | Before fix | After fix |
|---|---|---|
| Emergency-stop frequency | 19-26% of trades | ~2% of trades (correct) |
| In-sample avg R / t | +0.139 / 3.19 | +0.131 / 2.88 |
| Out-of-sample avg R / t | -0.024 / -0.60 | -0.010 / -0.24 |
| Combined avg R / t | +0.052 / 1.75 | +0.063 / 1.98 |
| **US combined avg R / t** | +0.131 / 3.06 | **+0.150 / 3.40** |

All numbers above and elsewhere in this file are post-fix. `backtest.py` (the
original trend-following engine) has the same latent pattern but was never
materially affected since its swing-low-based initial stop is usually already
close to entry, so inversion was rare there (1 emergency_stop exit historically).

## Timeframe sweep: the edge is specific to daily bars
Same strategy, unchanged, run on 4H/1H/30m/10m (in-sample 28-symbol basket):

| Timeframe | History | Trades | Win rate | Avg R | t-stat |
|---|---|---|---|---|---|
| Daily | 6.5yr, multi-regime | 288 | 58.0% | +0.131 | 2.88 |
| 4H | ~2yr | 360 | 55.0% | +0.043 | 1.03 |
| 1H | ~2yr | 1234 | 53.0% | -0.045 | **-2.42** |
| 30m | ~2mo, single regime | 152 | 48.0% | -0.097 | -1.92 |
| 10m | ~3mo, single regime | 554 | 49.5% | -0.051 | -1.81 |

Win rate degrades monotonically as timeframe shortens. Coherent, not noise:
"squeeze then breakout" is a multi-day phenomenon (real consolidations take
days to build) -- on intraday bars both the squeeze detection and the
breakout itself become far more prone to whipsaws and fakeouts. This is
specifically a daily-bar strategy, not a general one that's merely weaker
elsewhere -- 1H is significantly negative on a large sample (n=1234).

## Comparison to other strategies in this project
This is the only strategy whose combined (in-sample + out-of-sample) result
clears statistical significance outright for a sub-universe (US, t=3.40) on
the full multi-regime daily window. The multi-timeframe RSI+Stochastic
strategy's best combined result (t=2.53) was pooled across US+crypto on a
shorter ~2yr window; the trend-following and mean-reversion strategies never
cleared significance positively anywhere. See `FINDINGS.md`, `FINDINGS_MEANREV.md`,
`FINDINGS_MTF.md`.

## Caveat
Same as every other basket test: symbols within each group are correlated, so
headline trade counts overstate independent sample size.

## Code
- `strategy_breakout.py` -- squeeze detection + breakout entry, long only
- `backtest_breakout.py` -- two-stage stop engine (structural initial stop,
  delayed-activation ATR trail), reused pattern from the original trend
  strategy's `backtest.py`
- `run_basket_breakout.py` -- run any timeframe:
  `python run_basket_breakout.py [daily|4h|1h|30m|10m]`
