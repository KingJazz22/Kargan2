# XGBoost vs random forest meta-labeler — pre-registered spec

Written 2026-10-19, before any XGBoost result was computed. This is a design
comparison on trades that have already been used (see SPEC_RF_GROUPED.md),
so the only decision it can make is which model the live shadow log records.
It cannot approve a live filter.

## Setup
Identical to SPEC_RF_GROUPED.md after amendment 2:
- All 9,861 swing_pcse_exit trades.
- Yearly walk-forward over 2019–2026. Each year trains only on trades that
  exited before 1 January.
- The filter skips trades below the 20th percentile of in-sample
  probability, with a month-block bootstrap p.

## Models (hyperparameters fixed, never tuned)
- `rf_A` and `rf_B`: the random forest on the 16 base features (A), and on
  the base plus group features (B). B is the model currently in the shadow
  log.
- `xgb_A` and `xgb_B`: XGBClassifier(n_estimators=300, learning_rate=0.03,
  max_depth=3, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
  reg_lambda=1.0, tree_method="hist", random_state=0), with
  scale_pos_weight = losers/winners in the training window. Same A and B
  feature sets.

## Decision
`xgb_B` replaces `rf_B` in the shadow log only if both hold:
- it beats `rf_B`'s AUC in **at least 6 of 8** test years;
- its filter Δ is higher than `rf_B`'s.

Otherwise the shadow log keeps `rf_B`. `xgb_A` vs `rf_A` is reported for
information only.

## Amendments
(none)
