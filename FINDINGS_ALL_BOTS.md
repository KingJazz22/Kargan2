# Random-entry benchmark and honest buy-and-hold comparison for every live bot — Findings

**Conclusion.**
- **PCSE is the only strategy whose entry beats random entries:** +0.19R vs
  +0.16R per trade, with random below real in 10 of 10 draws. It is also
  the best strategy on its own (Sharpe 1.15, 12.7% CAGR vs 8.9% for SPY at
  the same exposure).
- **Breakout Hunter and Swing-Structure Breakout entries are worse than
  random entries** run through their own exits, in 10 of 10 draws, measured
  either in R or in percent return.
- **v2 book** (swing_pcse_exit + PCSE): 16.7% CAGR, Sharpe 1.17, max DD
  −23.3%. That beats SPY (15.1%, 0.90, −33.8%) and SPY at the same exposure
  (13.3%).
- **v1 book** (Breakout Hunter + Swing-Structure + PCSE): 15.3% CAGR, Sharpe
  0.94, max DD −27.6%, which is about SPY.

## 1. Random entries through each strategy's own exit
`random_entry_all_bots.py` swaps only the entry days. Every stop, trail,
target and sizing rule comes from the real engine. Ten random draws each,
with the same signal count per symbol segment, on random allowed days
(point-in-time S&P membership for fresh names).

| Strategy (bot) | Universe | Real trades | Real avg R | Random avg R (10 draws) | Random beats real | Verdict |
|---|---|---|---|---|---|---|
| **PCSE** (v1, v2) | 20 symbols, 4H, 2017–2026 | 492 | **+0.190** (t 5.4) | +0.156 (0.129…0.177) | **0/10** | entry adds about +0.03R |
| Breakout Hunter (v1; v2 entries off) | S&P point-in-time, daily | 5,530 | +0.026 (t 2.5) | **+0.091** (0.071…0.109) | **10/10** | entry worse than random |
| Swing-Structure, trailing exit (v1) | S&P point-in-time, daily | 13,134 | +0.040 (t 5.5) | **+0.076** (0.059…0.088) | **10/10** | entry worse than random |
| swing_pcse_exit (v2) | same | 9,861 | +0.093 | +0.103 | 20/20 | FINDINGS_RANDOM_ENTRY.md |

Robustness check in plain percent return, since a random entry gets a
different swing-low stop than a breakout entry:

| Strategy | Entries | Avg return per trade | Avg stop distance | Return per 100 days held |
|---|---|---|---|---|
| Breakout Hunter | real | +0.33% | 10.3% | +1.0% |
| Breakout Hunter | random (3 draws) | +0.61…0.68% | 8.1% | +2.3…2.5% |
| Swing-Structure | real | +0.48% | 11.4% | +1.4% |
| Swing-Structure | random (3 draws) | +0.57…0.67% | 8.2% | +2.1…2.5% |

Breakouts come with wider stops and still earn less per trade and per day
than random entries.

## 2. Each book and strategy vs buy-and-hold (honest daily marks, 2% risk, live breaker rule, $9,995 start)
| Book | Avg invested | CAGR | Sharpe | Max DD | SPY CAGR / Max DD | SPY at the same exposure | Beta |
|---|---|---|---|---|---|---|---|
| Breakout Hunter alone (40 symbols) | 39% | 6.3% | 0.83 | −17.5% | 15.1% / −33.8% | 6.0% | 0.14 |
| Swing-Structure alone (40) | 87% | 12.2% | 0.78 | −26.7% | 15.2% / −33.8% | **13.4%** | 0.69 |
| **PCSE alone (20)** | 58% | **12.7%** | **1.15** | −17.9% | 15.0% / −33.8% | 8.9% | 0.36 |
| swing_pcse_exit alone (40) | 76% | 13.3% | 1.08 | −24.8% | 15.1% / −33.8% | 11.6% | 0.59 |
| **v1 book** (BH + Swing-Structure + PCSE) | 92% | 15.3% | 0.94 | −27.6% | 15.2% / −33.8% | 14.0% | 0.75 |
| **v2 book** (swing_pcse_exit + PCSE) | 87% | **16.7%** | **1.17** | **−23.3%** | 15.1% / −33.8% | 13.3% | 0.66 |

- Swing-Structure with its trailing exit trails SPY at the same exposure.
  Breakout Hunter only matches it.
- PCSE adds the most per unit of exposure. It is also the least correlated
  with the market (correlation 0.60, beta 0.36), so it diversifies the
  others.
- **v1 vs v2:** they share PCSE. The difference is v2's swing_pcse_exit
  against v1's Breakout Hunter + trailing Swing-Structure. v2 wins on CAGR,
  Sharpe and drawdown.

## Not covered
- **Orland EWZ** (v1 side budget): there is no backtest engine, only the live
  strategy code, and it trades a single symbol.
- **v3 ML flat long-short** (paper): it is market-neutral, and its own study
  already compared it against all flat stocks equal-weighted. It is a
  forward test by design.
- v1's partial profit-taking on Breakout Hunter is not modeled.
- The Breakout Hunter and Swing-Structure engines charge no costs; PCSE and
  swing_pcse_exit charge 5 bps per side. Costs would lower the first two
  slightly.
- Taxes are ignored, and idle cash earns 0% (see FINDINGS_BUY_HOLD.md).

## Implications
1. **v1's Breakout Hunter and trailing Swing-Structure add nothing an index
   fund wouldn't.** v2 already disabled Breakout Hunter entries for this
   kind of reason. The case for v1 running them is now weak.
2. **PCSE is worth keeping in both bots.** It is the one entry with
   information in it, and its diversification is what makes the v2 book
   beat SPY.
3. ML filters on Breakout Hunter or Swing-Structure entries are pointless:
   random entries already do better. PCSE is itself an ML-probability entry.

Code: `random_entry_all_bots.py`, `compare_bots_buy_hold.py`. Output in
`research_metalabel/results_random_entry_all_bots.txt` and
`results_bots_buy_hold.txt`.
