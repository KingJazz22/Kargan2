# swing_pcse_exit vs random entries, plus a ledger look-ahead bug — Findings

**Conclusion: the breakout entry signal adds no edge.** Random entries with
the same exit (14×ATR stop, 3×ATR target, no time limit) on the same S&P
point-in-time universe did **better** than the real signals in all 20 of 20
draws: +0.103R vs +0.093R per trade, and 90.4% vs 89.5% wins. The strategy's
profit comes from two things, neither of them the signal:
1. **Barrier geometry.** With a stop 14×ATR away and a target 3×ATR away, a
   driftless random walk hits the target first 14/17 = **82.4%** of the
   time, with expected R = 0. The high win rate is built into the exits.
2. **Market drift.** US large caps rose strongly over 2016–2026, which lifts a
   random long entry from 82% to about 90% wins and from 0 to about +0.10R
   per trade.

So swing_pcse_exit is effectively **long US stocks with a wide stop and a
small, fixed profit-take**. It should do well in rising markets and poorly
in long bear markets, much like the market itself. It isn't a
signal-driven edge. This also explains why every meta-labeler found almost
nothing (AUC ~0.53): there was little entry information to find.

Code: `random_entry_benchmark.py`. Output:
`research_metalabel/results_random_entry.txt`.

## Result (same engine, costs and universe as FINDINGS_10DAY.md's baseline)
| Entries | Trades | Win % | Avg R | R per 100 capital-days | Avg hold (days) |
|---|---|---|---|---|---|
| Driftless random walk (theory) | — | 82.4% | 0.000 | — | — |
| **Real structure-confirmed breakouts** | 9,861 | 89.5% | **+0.093** | 0.091 | 102 |
| Random entries, 20 draws (mean) | ~13,470 | 90.4% | **+0.103** | 0.102 | ~101 |
| Random entries, worst draw | 13,478 | 90.2% | +0.100 | 0.099 | 101 |

- Each random draw puts the same number of signals into each symbol segment,
  on uniformly random allowed days: past warmup, and on point-in-time S&P
  membership days for fresh names.
- Random draws produce more trades because random days collide less with an
  already-open position.

## Ledger look-ahead bug (`sweep_risk_levels.replay_shared_ledger`, since commit 73ba1ac)
`equity_now()` values each open position at `notional × (1 + pnl_pct)`,
where `pnl_pct` is the trade's **final** result. From the day a position
opens, equity already reflects how the trade will end. Consequences:
- Max drawdown and Sharpe are **understated** everywhere the ledger was used.
  A position heading for a −1R stop shows no loss until the stop is hit, and
  winners show their gain on the first day.
- Position sizing uses future information: `risk_amount = equity × risk%`
  includes open trades' eventual P&L.
- The circuit-breaker and daily-loss checks reacted to future values.
- Final return is roughly right but path-dependent through the sizing.

Fixing it needs a daily mark-to-market from the price path of every open
position. Until then, read ledger CAGR as approximate and ledger max DD and
Sharpe as optimistic.

## What it means for live trading
- Expect swing_pcse_exit to track the US stock market's direction. Judge it
  against simply holding an equal-weight or SPY position with the same
  capital, not against zero.
- Improving the **entry** (ML filters, ranking, grouped models) has little to
  work with. Any real improvement has to come from exposure, exits, costs or
  risk control.
