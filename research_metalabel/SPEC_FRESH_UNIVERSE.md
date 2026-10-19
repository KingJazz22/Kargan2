# swing_pcse_exit on a fresh S&P 500 universe — pre-registered spec

Written 2026-10-19, before any result was computed. Any later change goes
under "Amendments". The ATR% rule in H2 came from looking at the old
104-symbol trades (FINDINGS_METALABEL follow-up), which is exactly why it is
tested here on names it was not derived from.

## Data
- `research_rotation_data/alpaca_bars/all` (Alpaca SIP daily bars, split +
  dividend adjusted), 2016-01 → 2026-10-16. That is 502 current and 255
  former S&P 500 members, delisted names included, plus ETFs.
- Duplicate listings: a bar with the same date, close and volume as a bar of
  another ticker is the same security. Keep one copy: current tickers before
  former ones, and the old 104 symbols before both. The kept copy counts as
  a member if either ticker was a member that day.
- Segments: a gap of more than 10 trading days, or a one-bar move beyond ×4
  or ÷4, starts a new segment. A trade still open at a segment's end exits at
  that segment's last close (delisting). A trade still open at the end of the
  data is dropped.
- **Old set** = the 104 symbols used in every earlier swing_pcse test, on all
  dates. **Fresh set** = every other stock, with signals taken only on days it
  was a point-in-time S&P 500 member.

## Strategy (unchanged from the deployed config)
Same as SPEC.md: the structure-confirmed breakout signal, entry at the next
open, a fixed stop at 14×ATR and a fixed target at 3×ATR, daily bars,
stop-first on a bar that touches both, 5 bps per side, one position per
symbol.

## H1: replication (does the edge exist on unseen names?)
Pass if all of these hold on the fresh set:
- mean R > 0 with t ≥ 3 (t over trades);
- mean R > 0 in both 2016–2020 and 2021–2026;
- mean R > 0 on trades from names that are no longer S&P members (the
  survivorship check). This one is reported, but because the sample is small
  it only fails H1 if mean R < 0.

The old-set numbers from this same engine are reported next to the fresh
ones for comparison.

## H2: capital efficiency (fresh set, 2%-risk shared ledger, `sweep_risk_levels.replay_shared_ledger`)
- Baseline: every fresh-set signal in time order. Same-day ties are broken
  by symbol order.
- `rank`: same-day ties are taken highest ATR% first.
- `floor`: skip signals with ATR% < 2.0%, roughly the old set's 40th
  percentile, fixed now.
- A rule passes if CAGR and Sharpe both beat the baseline on the full fresh
  set, in both time halves (each replayed separately), and in both symbol
  halves (fresh symbols split by the parity of `zlib.crc32(symbol)`), and if
  max DD is no worse than baseline − 3 points on the full set.
- Two rules are tested. A pass is a candidate for live use. A fail means
  neither rule ships.

## H3: ML meta-labeler transfer (train on old set, predict fresh set)
- Same 16 features as SPEC.md, all computed by this engine.
- Models, with fixed hyperparameters and no tuning:
  - `logit` and `lgbm`: as in SPEC.md.
  - `rf`: RandomForestClassifier(n_estimators=500, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced_subsample", random_state=0).
  - `mlp`: StandardScaler + MLPClassifier(hidden_layer_sizes=(32, 16),
    alpha=1e-3, early_stopping=True, max_iter=500, random_state=0).
- Each model is fit once on all old-set trades. The policy is `filter`: skip
  fresh trades whose probability is below the 20th percentile of the model's
  in-sample probabilities.
- A model passes if all of these hold:
  - Δ (R per trade, kept trades vs all fresh trades) > 0, with a
    month-block bootstrap one-sided p < 0.05/4 = 0.0125;
  - Δ > 0 in both time halves;
  - the 2%-risk ledger CAGR is not below the unfiltered fresh baseline.
- OOS AUC is also reported, but it is not part of the pass bar.

## Amendments
1. (Made after the first run, before any rule was changed.) The ledger's
   15% peak-drawdown circuit breaker latches. Once it trips, no new entries
   are taken, so equity can never recover to reset it. In the first run the
   fresh baseline tripped it in 2020 and took zero trades after that, which
   turned the H2 and H3 ledger comparisons into "which variant avoids
   tripping the breaker". The H2 and H3 ledger criteria are therefore judged
   with the breaker disabled (`--no-breaker`). The breaker-on results are
   kept in results_fresh.txt and reported as an operational finding. Nothing
   else changed: same rules, thresholds and models.
