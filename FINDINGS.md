# Adaptive Trend Following — backtest finding

**Conclusion: no demonstrated edge at any timeframe tested (daily down to 10m), for this symbol set, as specified.**

## Spec tested
EMA50/EMA200 trend filter, ADX(14)>=25 (recent, not same-bar — see below), pullback
to EMA50, reclaim candle, volume confirmation. Structural initial stop (swing low -
0.5xATR), ATR trailing stop (2.5xATR, activates after 1xATR of profit), fast reversal
exit (close beyond EMA50), ADX<20 hard kill, no fixed take profit. Full rules in
`strategy.py`.

## Sample
115-118 pooled trades across 20 US large-cap/ETF symbols + 8 crypto pairs,
2019-01-01 to 2026-08-01, daily bars, $100k starting equity, 1% risk/trade.

## Result
| Config | Trades | Win rate | Avg R | Profit factor | t-stat |
|---|---|---|---|---|---|
| Trail activates at 1.0x ATR | 118 | 33.1% | -0.15 | 0.64 | -1.72 |
| Trail activates at 0.5x ATR | 115 | 32.2% | -0.182 | 0.56 | **-2.16** |

Negative in both configs; earlier trail activation made it worse, not better.
Code is currently set to the 1.0x ATR config (the less-bad of the two).

## Why (exit breakdown, 1.0x config)
Only 12/118 trades (10%) ever reached the trailing stop. 78/118 exited via
`adx_kill` or `trend_reversal` before building a profit cushion. Trend-following
needs an asymmetric win/loss size to profit at a 33% win rate; this isn't getting it.

## Root-cause hypothesis (untested)
Two iterations tuning exit sensitivity (candle strictness, trail activation) both
failed to fix it — one made it worse. That points at the **entry**, not the exits:
buying a pullback to EMA50 after ADX has already been strong likely catches
late-stage moves rather than early ones. Not yet tested: earlier entry trigger
(e.g. EMA20 pullback), different confirmation logic.

## Timeframe sweep (daily strategy config, unchanged, run on 5 timeframes)
| Timeframe | History | Trades | Avg R | t-stat | Read |
|---|---|---|---|---|---|
| Daily | 6.5 yr, multi-regime | 118 | -0.15 | -1.72 | leaning negative — most trustworthy (longest, most diverse sample) |
| 4H | ~2 yr (Yahoo hourly cap) | 144 | -0.02 | -0.27 | flat/noise |
| 1H | ~2 yr (Yahoo hourly cap) | 491 | -0.004 pooled | -0.11 pooled | flat pooled; **crypto significantly negative (t=-2.65, n=160)**, US mildly positive not significant (t=1.33, n=331) |
| 30m | ~2 mo (Yahoo 30m cap), single regime | 68 | -0.18 | -2.84 | negative, but low-trust window |
| 10m | ~3 mo (Yahoo 5m cap), single regime | 152 | -0.12 | -2.99 | negative, but low-trust window; no commission/slippage modeled at this frequency |

No timeframe produced a convincing, trustworthy positive edge. The longest/most
diverse test (daily) leans negative with real signal behind it. The intraday
timeframes (30m/10m) are also negative but over too short and singular a window
to fully trust on their own. 1H's "flat" pooled result hides a statistically real
negative result specific to crypto. Going shorter than 10m only shrinks Yahoo's
usable history further (1m is capped at ~7 days) — not pursued.

## Caveat
The 20 US symbols are correlated (mostly move with the broad market), so 78 "US
trades" is not 78 independent bets — effective sample size is smaller than the
headline count. Same caveat applies at every timeframe above.

## Code
- `indicators.py` — EMA, Wilder ATR, Wilder ADX, volume SMA
- `strategy.py` — signal/exit rules, all thresholds as named constants at the top
- `backtest.py` — event-driven engine (signal on close, execute next open; stops fill intrabar)
- `data.py` — yfinance daily/1H/4H/30m/10m fetchers (4H and 10m are resampled since
  Yahoo has no native interval) + optional Alpaca (needs its own `.env` credentials,
  not the NewBOT account)
- `basket.py` — shared pooled multi-symbol backtest + significance-stat logic
- `run_backtest.py` — single-symbol daily run (SPY, BTC-USD) with equity curve PNG
- `run_basket.py` / `run_basket_4h.py` / `run_basket_1h.py` / `run_basket_30m.py` / `run_basket_10m.py` — pooled basket run per timeframe
