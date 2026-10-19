# swing_pcse entries with a max-10-day exit — Findings

**Conclusion: no.**
- Every 10-day exit (16 configs) loses to the current exit (14×ATR stop,
  3×ATR target, no time limit) in the cash-limited account. The best one made
  9.2% CAGR, against 15.2% for the current exit.
- Per trade, the 10-day versions keep only +0.002 to +0.024R, against
  +0.093R. That is a 0.0–0.17% average move, which is the same size as the
  10 bp round-trip cost.
- This entry's edge only shows up with a patient exit.
- By the pre-registered rule, the non-S&P holdout was **not** run. It is still
  unused.

Spec: [research_metalabel/SPEC_10DAY.md](research_metalabel/SPEC_10DAY.md).
Code: `backtest_10day.py`. Output: `research_metalabel/results_10day_dev.txt`.

## Development result (S&P point-in-time, 2016–2026, 2%-risk ledger with the live breaker rule)
| Exit | Trades | Win % | Avg R | t | Avg hold (days) | Time-stopped | Avg move/trade | Ledger CAGR | Max DD |
|---|---|---|---|---|---|---|---|---|---|
| **Current: 14×ATR stop / 3×ATR target, no limit** | 9,861 | 89.5% | **+0.093** | 24.6 | 102 | 0% | +3.39% | **15.2%** | −15.4% |
| Best 10-day: 5×ATR stop / 3×ATR target | 18,919 | 53.4% | +0.016 | 5.2 | 9.9 | 83% | +0.17% | 9.2% | −19.7% |
| 5×ATR stop / 2×ATR target | 20,445 | 55.3% | +0.014 | 5.1 | 9.0 | 67% | +0.16% | 5.8% | −16.0% |
| 3×ATR stop / 3×ATR target | 19,177 | 53.0% | +0.020 | 4.1 | 9.4 | 72% | +0.11% | 5.0% | −19.0% |
| Tightest: 1.5×ATR stop / 1×ATR target | 29,075 | 58.8% | −0.009 | −1.7 | 4.3 | 11% | −0.04% | 0.7% | 0.0% |

The full grid is in the results file.

## Why it fails
- **The edge needs time.** In the current setup, trades still open on day 10
  are underwater on average (−0.025R), and 88% of them go on to hit target
  if held (see the hold-time diagnostic). A 10-day limit cuts off exactly the
  recoveries that make the money, and 67–83% of the 10-day trades end on the
  time stop.
- **Tight stops starve the account.** At 2% risk, a 1.5–3×ATR stop makes each
  position 25–70% of equity, so the account fits only one to three
  positions. That is why the tight configs took only 5–815 of their ~20,000
  signals in the ledger. It's the sizing trap from
  FINDINGS_BREAKOUT_PCSE_EXIT.md in reverse.
- **Costs eat the rest.** An average move of 0.1–0.17% per trade leaves almost
  nothing after 10 bps of round-trip costs and any real slippage.

## Implication
A max-10-day system needs a different **entry**, one whose edge plays out
within days. Retrofitting this entry won't work. Earlier short-horizon
attempts in this project failed: RSI multi-timeframe sync, range fade,
multi-timeframe crossing, pairs and weekly reversal, ML in flat markets.
