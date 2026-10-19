# swing_pcse_exit on 580 unseen S&P 500 names, ATR% capital rules, and RF/neural-net meta-labelers — Findings

**Conclusion.**
- **The edge is real on names it was never tuned on.** It made +0.081R per
  trade (t=18) on 7,683 point-in-time S&P 500 trades, positive in both
  2016–2020 and 2021–2026. That is about 40% smaller than on the 104 symbols
  every earlier test used (+0.137R from this same engine). Those 104 are
  today's largest companies, so their numbers are inflated by survivorship.
  **Plan on roughly +0.08R per trade live, not +0.14R.**
- **H1 still fails as pre-registered.** Trades in names that later dropped out
  of the S&P lost slightly (−0.009R, n=1,548, t=−0.7).
- **H2 fails.** Neither ATR% capital rule held up. The pattern seen on the old
  104 symbols was in-sample noise.
- **H3 fails.** None of the 4 meta-labelers passed. The random forest and
  LightGBM came closest; the neural net found nothing.
- **Operational risk:** on a broad universe, the latching 15% circuit breaker
  in the live bot would have stopped all new entries in 2020.

Spec: [research_metalabel/SPEC_FRESH_UNIVERSE.md](research_metalabel/SPEC_FRESH_UNIVERSE.md).
It has one amendment, about the circuit breaker, explained below. Code:
`backtest_pcse_fresh_universe.py`. Trades are in `_pcse_fresh_trades.pkl`;
output is in `research_metalabel/results_fresh*.txt`.

## Data
- Alpaca daily bars from `research_rotation_data`, 2016 → 2026-10: 502
  current and 255 former S&P 500 members, delisted names included.
  Duplicate listings were removed (FB is META's history, and so on). A trade
  still open when a name delists exits at its last close (55 trades).
- Fresh signals were only taken on days the stock was in the S&P 500. None of
  the 580 fresh names overlaps the old 104.
- Same engine and rules as the deployed strategy: stop at 14×ATR, target at
  3×ATR, 5 bps per side.

## H1: replication
| Set | n | Win % | Mean R | t |
|---|---|---|---|---|
| Old 104 (this engine) | 2,178 | 93.3% | +0.137 | 20.6 |
| **Fresh 580** | **7,683** | **88.4%** | **+0.081** | **18.1** |
| Fresh 2016–2020 | 4,000 | | +0.073 | 11.4 |
| Fresh 2021–2026 | 3,683 | | +0.089 | 14.3 |
| Fresh, names later dropped from the S&P | 1,548 | | −0.009 | −0.7 |

The survivorship row fails the pre-registered criterion, mean R ≥ 0. It is
partly built in: a stock gets dropped from the index *because* it fell. You
can't know in advance which names will be dropped, so the honest live
expectation is the point-in-time +0.081R.

## H2: capital efficiency (2%-risk shared ledger, breaker off)
| Rule | Full CAGR / Sharpe / maxDD | 2016–20 CAGR | 2021–26 CAGR | Symbol half A / B CAGR | Verdict |
|---|---|---|---|---|---|
| Baseline | 12.6% / 2.09 / −18.8% | 5.9% | 17.2% | 9.7% / 13.6% | |
| Rank by ATR% | 14.4% / 2.15 / −21.9% | 5.7% | 19.8% | 12.9% / 13.2% | FAIL (worse in 2016–20 and symbol half B; DD 3.1 points worse) |
| Floor ATR% ≥ 2% | 11.4% / 1.90 / −27.5% | 4.4% | 15.8% | 8.7% / 10.7% | FAIL (worse in every slice) |

The ATR% pattern I found on the old 104 symbols (high-ATR% trades earned
about 2.5× more per dollar-day) does not replicate. Good that it wasn't
shipped on the strength of the in-sample observation.

## H3: random forest, neural net and others (fit on the old 104, applied to the fresh 580)
| Model | OOS AUC | Losers skipped | Δ R/trade | p (needs < 0.0125) | Halves | Ledger CAGR / DD (base 12.6% / −18.8%) | Verdict |
|---|---|---|---|---|---|---|---|
| Logistic | 0.529 | 132/889 | +0.004 | 0.086 | +/+ | 13.3% / −15.6% | FAIL |
| LightGBM | 0.551 | 194/889 | +0.009 | 0.016 | +/+ | 14.0% / −17.1% | FAIL (p) |
| **Random forest** | 0.547 | 198/889 | +0.008 | 0.045 | +0.014 / +0.002 | **15.9% / −11.7%** | FAIL (p) |
| Neural net (MLP 32-16) | 0.506 | 182/889 | −0.002 | 0.72 | −/+ | 13.6% / −22.3% | FAIL |

- **This is the strongest ML result in the project, and it still isn't
  proven.** Trees trained on one set of stocks and applied unchanged to 580
  others kept a small positive edge, and the random forest's ledger looks
  clearly better.
- **What keeps it from passing:** p=0.045 doesn't clear the 4-trial
  correction, nearly all of the gain is in 2016–2020, and a single ledger
  path is fragile.
- **The neural net learned nothing transferable.** That fits about 2,000
  noisy training rows, which is far too few for a neural net to beat trees.

## Operational finding: the latching circuit breaker
With the 15% peak-drawdown breaker on, as in `live_trading_v2.py`, the fresh
baseline tripped it during COVID in 2020. It then never took another trade,
for a 1.6% CAGR instead of 12.6%. The live bot works the same way: it halts
new entries until equity gets back within 15% of its peak, and only
positions it already holds can get it there. The old 104 never got deep
enough to trip it (max DD −11.5%). A broader or less lucky universe can.
Decide in advance how the breaker gets reset, for example automatically
after N days, or a manual review with a known procedure.

## What to do with this
1. Keep swing_pcse_exit live, but expect about +0.08R per trade.
2. Don't ship either ATR% rule.
3. Optional: shadow-log the random forest's probability for each live signal,
   recording it with no effect on orders, then judge it on 6–12 months of
   forward data. No more backtest tuning; this data is now spent for
   meta-labeling.
4. Decide on the circuit-breaker reset policy before the next deep drawdown.
