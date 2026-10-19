# Meta-labeling swing_pcse_exit — pre-registered spec

Written 2026-10-19, before any result was computed. Any later change is listed
under "Amendments" with its reason.

## Question
Can a secondary ML model, given only information known at the signal close,
tell which swing_pcse_exit entries will lose (hit the 14×ATR stop before the
3×ATR target)? And does skipping or down-sizing those entries raise the
return per unit of risk?

## Primary strategy (unchanged, as deployed in live_trading_v2.py)
- Entry: `strategy_swing_breakout.prepare` `long_entry` on the daily close.
  Fill at the next open, plus 5 bps.
- Stop: entry − 14 × ATR(14) at signal. Target: entry + 3 × ATR at signal. Both
  fixed. Daily bars, gap-aware, stop checked first on a bar that touches both.
  Exit less 5 bps.
- One open position per symbol. No account-level cap here; that is the
  ledger check below.
- Data: split-adjusted daily cache (`split_adjust.load_daily_cache_adjusted`),
  TOP100 ∪ live-40 (104 symbols), 2016-01 → 2026-09. Warmup is 70 bars.

## Label and features
- R = (exit − entry) / stop distance. Meta-label y = 1 if R > 0.
- Features, frozen. All are computed on the signal bar, with no future data:
  atr_pct, adx14, rsi14, bb_z20, breakout_atr ((close − resistance)/ATR),
  range_atr ((resistance − support)/ATR), vol_ratio (vol / 20-day avg), ret5,
  ret20, ret60, dist_252h (close / 252-day high − 1), rv_ratio (20-day /
  60-day realized vol), spy_ret20, spy_above_sma200, spy_rv20, spy_adx14.

## Models (hyperparameters fixed, never tuned)
- `logit`: StandardScaler + LogisticRegression(C=1.0, class_weight="balanced").
- `lgbm`: LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7,
  min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
  is_unbalance=True, random_state=0).

## Validation: combinatorial purged CV
- Trades are sorted by entry date and cut into N=6 contiguous time groups.
  Each split tests k=2 groups: 15 splits, 5 backtest paths.
- Purge: drop every training trade whose [entry, exit] interval overlaps a
  test group's time span. Embargo: also drop training trades that enter
  within 10 trading days after a test group ends.
- Each trade gets 5 out-of-sample probabilities. A path uses one of them; the
  pooled score is their mean.

## Policies (4 trials = 2 models × 2 policies)
- `filter`: skip the trade if its OOS probability is below the 20th
  percentile of the in-sample (training-fold) probabilities.
- `size`: weight = 2 × the trade's percentile within the training-fold
  probabilities. The weight runs from 0 to 2 and averages ≈1.

## Metric and pass bar (all must hold, for a given model/policy)
1. Primary metric: R per unit of risk, sum(w·R)/sum(w). Δ = policy −
   unfiltered baseline (w ≡ 1).
2. Δ > 0 on all 5 CPCV paths.
3. On the pooled scores, Δ's one-sided p-value from a month-block bootstrap
   (5,000 resamples) is below 0.05/4 = 0.0125, Bonferroni-corrected for the 4
   trials.
4. Δ > 0 separately on the IS (`US_SYMBOLS`) and OOS (`OOS_US_SYMBOLS` + the
   rest of TOP100) symbol halves.
5. Dollar check (filter policy only): run the 2%-risk shared-ledger replay
   (`sweep_risk_levels.replay_shared_ledger`) on the pooled-score filtered
   trades. Total return must be at least the unfiltered return, and max
   drawdown no worse. A filter that raises R per trade but loses dollars
   fails; see the sizing trap in FINDINGS_BREAKOUT_PCSE_EXIT.md.

Also reported, but not part of the pass bar: OOS AUC per path, the
probabilistic Sharpe of the per-trade R stream, and feature importances.

## What happens on a result
- Pass: paper-trade the filter in live_trading_v2 as a shadow log (no order
  changes) for 3 months before going live.
- Fail: record it in FINDINGS_METALABEL.md and do not retune features or
  hyperparameters on this data.

## Amendments
(none)
