# swing_pcse_exit vs buy-and-hold, with an honest ledger — Findings

**Conclusion.**
- **Raw return:** valued honestly day by day, the live strategy (40 symbols)
  made **13.3% a year against SPY's 15.1%** over 2016–2026, so it trailed
  simply holding SPY.
- **Risk:** it carried less risk. Volatility was 12.3% vs 17.4%, max drawdown
  −24.8% vs −33.8%, and Sharpe 1.08 vs 0.90.
- **Like-for-like:** it was 76% invested on average. Against a 76% SPY / 24%
  cash portfolio, which is the fair comparison, it is ahead by 1.6 points a
  year with a higher Sharpe.
- **What it is:** about 0.6× the market (beta 0.59, correlation 0.84) with a
  modest edge from the exit structure. It is **not** bear-market protection:
  −18.9% in 2022 vs SPY's −18.2%, and −1.4% in 2020 vs SPY's +18.5%.

## The ledger fix (`sweep_risk_levels.replay_shared_ledger(..., marks=...)`)
- **The bug:** the old ledger valued every open position at its *final*
  result from the day it opened (FINDINGS_RANDOM_ENTRY.md).
- **New mode:** passing `marks={symbol: daily closes}` instead marks each
  open position to the last close known before each event. An entry at day
  D's open is sized off D−1's close. Equity is also recorded at every close,
  together with the invested capital.
- **Old mode:** without `marks` the old behavior is unchanged, so earlier
  findings still reproduce (regression check: +324.0% / −11.5%, exact).
- A first run gave nonsense: 5% volatility and a 0.22 correlation for a
  97%-invested long book. The cause was pandas 3 storing these dates in
  microseconds; it is fixed.

How wrong the old ledger was, on the live 40 symbols:

| | CAGR | Volatility | Sharpe | Max DD |
|---|---|---|---|---|
| Old look-ahead ledger | 16.3% | 7.9% | **4.63** | **−9.9%** |
| **Honest daily marks** | **13.3%** | **12.3%** | **1.08** | **−24.8%** |

Every ledger drawdown and Sharpe in earlier findings files is optimistic by a
similar margin.

## Result: $9,995 start, 2% risk, live breaker rule, dividend-adjusted prices, 2016-04 → 2026-10
**Live 40 symbols** (what `live_trading_v2` trades), 76% invested on average:

| Portfolio | Total return | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| **swing_pcse_exit** | +270% | **13.3%** | 12.3% | **1.08** | **−24.8%** |
| SPY buy & hold | +337% | **15.1%** | 17.4% | 0.90 | −33.8% |
| RSP (equal-weight S&P) buy & hold | +225% | 11.9% | 18.0% | 0.72 | −39.1% |
| 76% SPY + 24% cash (same exposure) | +218% | 11.6% | 13.3% | 0.90 | −26.7% |

**All point-in-time S&P names** (the broader universe), 97% invested on
average:

| Portfolio | Total return | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| **swing_pcse_exit** | +342% | 15.2% | 15.3% | 1.00 | −26.6% |
| SPY buy & hold | +333% | 15.0% | 17.4% | 0.89 | −33.8% |
| 97% SPY + 3% cash | +319% | 14.6% | 17.0% | 0.89 | −33.1% |

Calendar years, live 40 vs SPY, %:

| | 2016* | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Strategy | 9.3 | 32.5 | 5.3 | 30.9 | −1.4 | 20.0 | −18.9 | 16.5 | 22.8 | 13.1 | 19.6 |
| SPY | 9.3 | 21.7 | −5.0 | 31.1 | 18.5 | 28.6 | −18.2 | 26.2 | 24.9 | 17.7 | 14.1 |

\*partial years. The broad universe's 2026 (+66%) is concentrated in a few
very volatile names (MRNA +92% in 8 days, SNDK, INTC, NTAP). Those are real
moves in the data, but a lot of luck sits in one partial year.

## The circuit breaker, re-checked with honest marks (2% risk)
| Rule | Live-40 CAGR / Sharpe / Max DD | S&P PIT CAGR / Sharpe / Max DD |
|---|---|---|
| Latch (old live rule) | **5.7%** / 0.70 / −18.5% | **3.2%** / 0.40 / −23.6% |
| No breaker | 15.4% / 1.16 / −26.1% | 15.1% / 0.99 / −27.0% |
| **Re-arm 30 days, half risk (deployed)** | 13.3% / 1.08 / −24.8% | 15.2% / 1.00 / −26.6% |
| Re-arm 30 days, full risk | 15.8% / 1.18 / −26.0% | 14.3% / 0.95 / −28.9% |

- With honest marks, the **old latch would have halted the live 40 symbols
  too**, blocking 491 entries for 5.7% CAGR. The biased ledger hid this,
  because its drawdown never reached 15%.
- The deployed re-arm rule fixes it. On the live 40 it costs about 2 points
  of CAGR against no breaker or a full-risk re-arm, in exchange for slightly
  smaller drawdowns. On the broad universe it is the best variant. Which to
  run is a risk preference; the deployed one is the more cautious choice.

## Caveats
- Idle cash earns 0% here. T-bills paid 4–5% in 2023–2025, which would add
  roughly 1 point a year to the strategy at 76% invested, and to the
  matched benchmark too.
- Costs are 5 bps per side. **Taxes are ignored.** Nearly all of the
  strategy's gains are short-term (about 100-day holds), while SPY
  buy-and-hold defers tax. In a taxable account that gap is large.
- Entries are the backtest's, on dividend-adjusted daily bars, not
  tick-precise fills.

Code: `compare_buy_hold.py`, `sweep_risk_levels.py` (marks mode). Output:
`research_metalabel/results_buy_hold.txt`.
