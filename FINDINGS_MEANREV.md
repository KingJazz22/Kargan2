# Mean Reversion — backtest finding

**Conclusion: no demonstrated edge at any timeframe tested (daily down to 10m), for this symbol set, as specified.**

## Spec tested
ADX(14) range regime, entry on close below lower Bollinger(20,2) AND below lower
Keltner(EMA20, ATR10, 1.5x) AND RSI(14) oversold AND Stochastic(14,3,3) %K crossing
above %D from the oversold zone. Short is the mirror. Exit at the middle Bollinger
band OR an ATR(14) trailing stop (2x), whichever comes first. Full rules in
`strategy_meanrev.py`.

Two gaps in the original spec, filled in and flagged:
- "Price below Bollinger" + "price outside Keltner" read as both conditions must
  hold simultaneously (double-confirmation of an extreme), not a squeeze/expansion
  signal.
- No stop-loss was specified. Rather than invent a separate structural stop, the
  ATR trailing stop is active from bar 1 of the trade (not delayed like the trend
  strategy's), so it doubles as the protective stop -- appropriate since mean
  reversion expects a fast move back to the mean, not a running trend.

## Root-cause issue found and fixed
Requiring ADX<20 on the exact signal bar was nearly self-contradicting: RSI
oversold / BB+KC extremes are sharp-move signatures that push ADX UP, not down.
On SPY daily, 0 of 26 RSI-oversold bars ever had same-bar ADX<20 (avg ADX during
RSI<30 was 30.7 vs. 23.7 overall). Fixed the same way as the trend strategy's
ADX-timing conflict: "range regime" now means ADX was mostly under 25 over the
prior 10 bars, not on the exact touch bar.

## Timeframe sweep (unchanged strategy config, run on 5 timeframes)
| Timeframe | History | Trades | Win rate | Avg R | t-stat | Read |
|---|---|---|---|---|---|---|
| Daily | 6.5 yr, multi-regime | 148 | 37.8% | -0.096 | -1.29 | leaning negative, not significant |
| 4H | ~2 yr (Yahoo hourly cap) | 154 | 34.4% | -0.098 | -1.30 | leaning negative, not significant |
| 1H | ~2 yr (Yahoo hourly cap) | 447 | 37.4% | -0.066 pooled | -1.52 pooled | flat pooled; **crypto significantly negative (t=-2.50, n=302)**, US mildly positive not significant (t=0.69, n=145) |
| 30m | ~2 mo (Yahoo 30m cap), single regime | 78 | 42.3% | +0.056 | 0.63 | slightly positive, not significant, low-trust window |
| 10m | ~3 mo (Yahoo 5m cap), single regime | 212 | 41.5% | +0.016 | 0.37 | ~flat, not significant, low-trust window |

Exit mix at every timeframe: trailing stop fires roughly 2-3x more often than the
mean-reversion target is reached (e.g. daily: 99 stopped vs. 49 reached target).
The strategy is mostly getting stopped out, not mean-reverting successfully, even
where net R is near zero.

## Parameter sweep on 30m (looking for a better entry threshold)
Swept RSI oversold (20/25/30/35), Stochastic oversold (15/20/25), Bollinger stdev
(1.5/2.0/2.5), and Keltner multiplier (1.0/1.5/2.0) -- 108 combos, run on cached
30m data (`sweep_meanrev_30m.py`, results in `sweep_meanrev_30m_results.csv`).

Found: the Keltner condition is nearly always redundant with Bollinger in this
data -- when price closes below the lower BB, it's already below the lower KC
96-100% of the time across kc_mult 1.0-2.0 (tested on SPY). KC_MULT had no
measurable effect on results across the grid.

Best combo (RSI<25, Stoch<15, BB_std=2.0): n=34, win rate 50%, avg R=+0.168,
t=1.10. 72% of the 72 combos with >=20 trades showed positive avg R, with a
sensible gradient (tighter/more extreme thresholds outperformed looser ones),
but **0/72 combos reached statistical significance (t>2)**.

## Out-of-sample check: does the best 30m combo hold on other timeframes?
Ran the winning combo (RSI<25, Stoch<15, BB_std=2.0) unchanged on daily/4H/1H/10m
(`test_combo_timeframes.py`). It does not generalize -- classic overfitting signature:

| Timeframe | Trades | Win rate | Avg R | t-stat |
|---|---|---|---|---|
| Daily | 59 | 30.5% | -0.286 | -2.82 (significantly worse than default params) |
| 4H | 68 | 35.3% | -0.124 | -1.25 |
| 1H | 186 | 33.3% | -0.116 | -1.64 |
| 30m (source window) | 34 | 50.0% | +0.168 | 1.10 |
| 10m | 96 | 39.6% | -0.017 | -0.27 |

The combo only works on the exact window it was optimized on and gets worse
everywhere else -- with 108 combinations tested against a small, single-regime
sample, some were going to look good by chance. This does not reflect a real
edge in the entry logic.

## Comparison to the trend-following strategy (same symbol set, same timeframes)
Mean reversion is consistently less bad than trend-following at every comparable
timeframe (higher win rate, less negative avg R, better profit factor) but neither
strategy has a demonstrated edge anywhere in the sweep. See `FINDINGS.md`.

## Corrected 2026-08-24: gap-through stop-fill bug fixed, daily re-run
`backtest_mr.py`'s trailing-stop exit filled at the theoretical stop price
even on bars that gapped straight through it -- same bug found project-wide,
fixed by clamping the fill to the bar's Open when it already gapped past the
stop (see `FINDINGS_SWING_STRUCTURE.md`'s correction note for the general
writeup). Re-ran the daily basket (`run_basket_meanrev.py daily`):

| | Before | After |
|---|---|---|
| Trades | 148 | 144 (AAPL skipped this run on a transient fetch timeout, not the fix) |
| Win rate | 37.8% | 38.2% |
| Avg R | -0.096 | -0.093 |
| t-stat | -1.29 | -1.22 |

Essentially unchanged -- this strategy already had no live position sizes
riding on a stop-loss edge case, since the whole point is it has no
demonstrated edge. **Verdict unchanged: no demonstrated edge at any
timeframe.**

## Caveat
Same as the trend-following findings: the 20 US symbols are correlated, so trade
counts overstate independent sample size. Same caveat applies at every timeframe.

## Code
- `indicators.py` — adds RSI, Stochastic, Bollinger Bands, Keltner Channel (plus the
  existing EMA/ATR/ADX/volume SMA)
- `strategy_meanrev.py` — signal/exit rules, thresholds as named constants
- `backtest_mr.py` — event-driven engine (signal on close, execute next open; ATR
  trailing stop active from entry, fills intrabar)
- `basket.py` — shared pooled multi-symbol backtest logic (`backtest_fn` parameter
  makes it reusable across strategies)
- `run_basket_meanrev.py` — run any timeframe: `python run_basket_meanrev.py [daily|4h|1h|30m|10m]`
