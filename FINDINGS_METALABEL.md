# Meta-labeling swing_pcse_exit — Findings

**Conclusion: meta-labeling doesn't help. All 4 pre-registered configs fail
(2 models × filter/size), and none reaches significance even before the
Bonferroni correction. Out-of-sample AUC is 0.51–0.52, so the 16 features
can't tell which entries will hit the 14×ATR stop. Keep trading
swing_pcse_exit unfiltered at 1× size.**

The spec, features, models and pass bar were written before any result
([research_metalabel/SPEC.md](research_metalabel/SPEC.md)). There were no
amendments.

## Setup (`backtest_metalabel_pcse.py`; output in `research_metalabel/results.txt`, trades in `_metalabel_trades.pkl`)
- Primary signal: the deployed swing_pcse_exit, unchanged. Structure-confirmed
  breakout entry, fixed stop at 14×ATR and fixed target at 3×ATR, daily bars,
  5 bps per side. Data is the split-adjusted daily cache: 104 symbols,
  2016-05 → 2026-09.
- 2,146 trades, 91.9% winners, mean +0.120R (t=16.3). The 174 stop-outs are
  what the model has to find.
- Label: win/loss. 16 features, all known at the signal close: ATR%, ADX, RSI,
  BB z-score, breakout and range size in ATR, volume ratio, 5/20/60-day
  returns, distance from the 52-week high, vol ratio, and SPY's 20-day return,
  SMA200 state, realized vol and ADX.
- Validation: combinatorial purged CV, N=6 groups, k=2 (15 splits, 5 paths).
  Purge on [entry, exit] overlap, plus a 10-day embargo.

## Result (Δ = change in R per unit of risk vs. unfiltered +0.1196R)
| Model / policy | OOS AUC | Δ pooled | Paths > 0 | Bootstrap p (needs < 0.0125) | IS / OOS symbols | 2%-risk ledger | Verdict |
|---|---|---|---|---|---|---|---|
| logit / filter | 0.521 | +0.0011 | 4/5 | 0.39 | −0.007 / +0.004 | +241% vs +324%, DD −14.0 vs −11.5 | FAIL |
| logit / size | 0.521 | +0.0050 | 4/5 | 0.089 | −0.001 / +0.007 | — | FAIL |
| lgbm / filter | 0.514 | +0.0079 | 4/5 | 0.10 | +0.006 / +0.009 | +341% vs +324%, DD −14.0 vs −11.5 | FAIL |
| lgbm / size | 0.514 | +0.0015 | 2/5 | 0.33 | −0.002 / +0.002 | — | FAIL |

- The logit filter skipped 20% of trades but only 36 of the 174 losers. That
  is about what a random 20% skip would catch (35). The win rate was 92.0%
  among kept trades and 91.5% among skipped ones.
- The LightGBM filter is the closest call: +0.008R, and dollar return +341%
  vs. +324%. But its drawdown was worse (−14.0% vs −11.5%), one path is
  negative, and p=0.10. That fails 3 of the 4 criteria.
- LightGBM's top features were SPY realized vol, ATR% and ADX. That is the same
  pattern as in FINDINGS_ML_FLAT.md: the model learns which trades are
  volatile, not which ones win.

## Sanity check
I added a weak artificial leak of the label into one feature (`atr_pct`) and
re-ran the same CPCV. It gave an OOS AUC of 0.83, so the pipeline does detect
signal when signal exists. The null result is not a bug.

## How to read it
- With 92% winners, a 14×ATR stop and a 3×ATR target, the stop-outs look like
  scattered tail events. Loss rates by year run from 4% to 13%, and the high
  years are 2018 and 2021. A per-trade classifier gets no grip on them from
  price or volume features.
- This matches the earlier failures of the EMA34 regime filter and the HMM
  idea. Conditioning on regime, volatility or trend state doesn't separate
  this strategy's winners from its losers.
- Don't retune features or hyperparameters on this data; that would just fit
  the noise. A genuinely new data source, such as earnings dates
  (`_earnings_dates_cache.pkl` exists) or news, would be a separate
  pre-registered test.
