# Random forest: one pooled model vs models split by stock / liquidity / sector / regime — pre-registered spec

Written 2026-10-19, before any result was computed. This is a **design
comparison**, not a validation. The trades were already used in
FINDINGS_PCSE_FRESH_UNIVERSE.md, so whichever design wins only becomes a
forward-test (shadow-log) candidate, never a live filter on this evidence.

## Data
- All 9,861 swing_pcse_exit trades from `_pcse_fresh_trades.pkl`: the old
  104 + 580 fresh point-in-time S&P names, 2016–2026, 1,035 of them losers.
- Group variables, all point-in-time at the signal close:
  - `liq_pct`: cross-sectional percentile of the 63-day median dollar volume
    among all stocks with bars that day. There is no shares-outstanding data,
    so this one variable stands in for both size and liquidity.
    Terciles: low / mid / high.
  - `sector`: whichever of the 11 SPDR sector ETFs has the highest 252-day
    daily-return correlation with the stock. Inferred because the repo has no
    sector data.
  - `regime`: SPY above or below its 200-day SMA.
  - `stock`: the ticker.

## Designs (RF hyperparameters as in SPEC_FRESH_UNIVERSE.md, 16 base features)
- **A pooled**: one model on all trades.
- **B pooled + group features**: A plus `liq_pct`, a one-hot `sector`, and the
  stock's own loss rate. The loss rate is computed from training-fold
  trades only and shrunk toward the global rate with a prior of 20 trades.
- **C per liquidity tercile**: 3 separate models.
- **D per sector**: one model per sector. A sector falls back to A when its
  training fold has fewer than 30 losers.
- **E per regime**: 2 models.
- **F per stock**: one model per stock with at least 40 training trades and
  at least 5 training losers; every other stock falls back to A. The count of
  stocks that qualify is reported.
- **G blend**: the mean of A, C, D and E's probabilities (the "every label"
  hierarchy).

## Validation and decision
- CPCV with N=6 and k=2 (15 splits), purge plus a 10-day embargo
  (`backtest_metalabel_pcse.cpcv_splits`).
- Reported per design: mean OOS AUC; how many of the 15 splits beat A's AUC;
  and the filter policy (skip trades below the 20th percentile of in-sample
  probability) Δ R per trade on the path-averaged probabilities, with a
  month-block bootstrap p.
- A grouped design (B–G) **helps** if it beats A's AUC in at least 12 of 15
  splits **and** its filter Δ is higher than A's. If none does, grouping
  doesn't help, and pooled A stays the only candidate.

## Amendments
1. (After the first run crashed on split 3, with no results read beyond
   splits 1 and 2.) Purging against each test group's whole
   [first entry, last exit] span left an empty training set, because 527
   trades are held more than a year (the longest 5 years). Switched to exact
   purging (`cpcv_splits(exact=True)`): drop only training trades whose own
   [entry, exit] overlaps a test trade's, and embargo 10 days after each test
   group's last entry. Designs, models and the decision rule are unchanged.
2. (Exact purging still left split (0,3) with 0 training trades, because
   multi-year holds overlap nearly every period. No AUC beyond splits 1–2 had
   been seen.) Replaced CPCV with the constraint live trading actually has:
   **yearly walk-forward**. For each test year from 2019 to 2026, train on
   every trade that *exited* before 1 January of that year, so its label was
   known, and test on the trades entered during that year. The decision rule
   becomes: beats A's AUC in **at least 6 of 8 test years** and has a higher
   filter Δ than A. Everything else is unchanged.
