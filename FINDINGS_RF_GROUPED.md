# Random forest: one pooled model vs models split by stock / liquidity / sector / regime — Findings

**Conclusion.**
- **Splitting the data into separate models makes things worse.** It's the
  overfitting you suspected.
  - Per-stock models are impossible: no stock had even 5 losing trades in a
    training window.
  - Per-sector models were the worst design (mean AUC 0.497, worse than a
    coin flip).
  - Per-liquidity and per-regime models, and the blend of all of them, all
    came in below the single pooled model.
- **Giving one pooled model the group information as extra inputs** (a
  liquidity percentile, the sector, and the stock's own shrunk loss rate) was
  the best design. Mean AUC 0.554 vs 0.534 pooled; filter Δ +0.010R per
  trade, p=0.009.
- It beat the pooled model in only 5 of 8 years, one short of the
  pre-registered bar, so **nothing passes**. This is a design comparison on
  already-used trades, so the pooled-plus-group-features model is at best a
  shadow-log candidate.

Spec: [research_metalabel/SPEC_RF_GROUPED.md](research_metalabel/SPEC_RF_GROUPED.md),
which has 2 amendments, both explained below. Code:
`backtest_rf_grouped.py`. Output: `research_metalabel/results_rf_grouped.txt`.

## Setup
- All 9,861 swing_pcse_exit trades: the old 104 + 580 point-in-time S&P
  names, 2016–2026, 1,035 of them losers.
- Liquidity: 63-day median dollar volume, as a cross-sectional percentile.
  It also stands in for size, because there is no shares-outstanding data.
- Sector: inferred point-in-time as the SPDR sector ETF with the highest
  252-day return correlation. The repo has no sector data.
- Regime: SPY above or below its 200-day SMA.
- Validation: yearly walk-forward over 2019–2026. Each year's models train
  only on trades that had **already closed** before 1 January, which is the
  same constraint live trading has.

## Why the validation changed (amendments 1–2)
These trades are held a median of 47 days, but 527 of them ran more than a
year and the longest ran 5 years. Purged combinatorial CV removed every
training trade that overlapped a test trade, and for some splits that left
**zero** training trades. That forced the switch to walk-forward.

It also casts doubt on the earlier "random forest nearly passes" result in
FINDINGS_PCSE_FRESH_UNIVERSE.md H3. That test trained on old-set trades from
the *same years* it tested on, so shared market-wide moves may have
inflated it. In this walk-forward the pooled RF still shows something (Δ
+0.0095R, p=0.02), but with a mean AUC of only 0.534.

## Result (walk-forward 2019–2026, 6,916 test trades)
| Design | Mean AUC | Years beating A (needs ≥ 6/8) | Filter Δ R/trade | p | Verdict |
|---|---|---|---|---|---|
| A: one pooled model | 0.534 | — | +0.0095 | 0.021 | baseline |
| **B: pooled + group features** | **0.554** | 5 | **+0.0099** | **0.009** | no (5/8) |
| C: per liquidity tercile | 0.528 | 4 | +0.0039 | 0.125 | no |
| D: per sector | 0.497 | 0 | +0.0021 | 0.287 | no |
| E: per regime | 0.527 | 4 | +0.0099 | 0.016 | no |
| F: per stock | 0.534 | 0 | +0.0095 | 0.026 | no (0 stocks qualified, so identical to A) |
| G: blend of A, C, D, E | 0.529 | 5 | +0.0073 | 0.058 | no |

AUC by year for A: 2019 0.58, **2020 0.45**, 2021 0.54, 2022 0.55, 2023
0.51, 2024 0.61, 2025 0.52, 2026 0.51. Every design scored below 0.5 in 2020.
What the model had learned before COVID turned upside down in the crash.

## How to read it
- **The data is thin where it matters.** There are 1,035 losers in total; the
  median stock has 1, and the median sector has about 100. Every split
  divides those few losers further, so each sub-model sees less signal and
  more noise. That's why C, D and G lose to A.
- **Group information helps most as an input.** B lets one model use all
  1,035 losers while still noticing patterns such as XLC/XLY/XLE names and
  low-liquidity names losing more often (13–16% vs 6–9%). That is the
  standard fix for this situation: pool the data, and pass the groups in as
  features or as shrunk loss rates.
- **Even the best design is worth about +0.01R per trade.** That is roughly a
  12% gain on the +0.08R base edge, and it isn't proven. It is a candidate to
  shadow-log alongside the live signals, not a filter to trade on.

## Follow-up: XGBoost (SPEC_XGB.md, `backtest_xgb_metalabel.py`, `research_metalabel/results_xgb.txt`)
**XGBoost does not beat the random forest, so the shadow log keeps the RF.**
Same walk-forward and trades as above. XGBoost 3.4.1, hyperparameters fixed
before the run.

| Model | Mean AUC | Filter Δ R/trade | p | Losers skipped |
|---|---|---|---|---|
| RF, base features | 0.534 | +0.0095 | 0.021 | 190/727 |
| **RF + group features (shadow model)** | **0.554** | **+0.0099** | **0.009** | 178/727 |
| XGBoost, base features | 0.527 | +0.0073 | 0.068 | 190/727 |
| XGBoost + group features | 0.546 | +0.0060 | 0.046 | 170/727 |

- XGBoost + group features beat the RF + group features in only 3 of 8 years,
  short of the 6/8 bar, and had a smaller filter gain.
- Base XGBoost scored **0.361** AUC in 2020, well below a coin flip and worse
  than the RF's 0.447. Boosting fits the pre-COVID patterns harder, so the
  crash flipped them harder.
- This matches the LightGBM result. Gradient-boosted trees (LightGBM,
  XGBoost) are not better than a random forest on about 1,000 noisy losers.
  Averaging many deep, independent trees (RF) is the more robust choice at
  this sample size.
