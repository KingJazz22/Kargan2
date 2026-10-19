# swing_pcse entries with a max-10-day exit — pre-registered spec

Written 2026-10-19, before any 10-day result was computed.

## Question
Can the same structure-confirmed breakout entries, paired with exits built
for a 10-day horizon, beat the current exit (14×ATR stop, 3×ATR target, no
time limit) in the cash-limited account? Faster exits free cash about 10×
sooner, so the account can take more of the signals it currently skips.

## Stage 1: development (S&P point-in-time, the same universe and engine as SPEC_FRESH_UNIVERSE.md)
- Entry is unchanged: signal on the daily close, fill at the next open plus
  5 bps, one position per symbol.
- Exit: fixed stop at SL×ATR and fixed target at TP×ATR (ATR at the signal),
  checked on daily bars, gap-aware, stop first. Any trade still open on the
  first bar ≥ 10 calendar days after entry exits at that bar's close. 5 bps
  per side.
- Grid: TP ∈ {1.0, 1.5, 2.0, 3.0} × SL ∈ {1.5, 2.0, 3.0, 5.0} (16 configs),
  plus the deployed baseline (SL 14, TP 3, no time limit).
- Account: the 2%-risk shared ledger (`sweep_risk_levels.replay_shared_ledger`)
  with the live breaker rule (re-arm after 30 days, half risk), starting at
  $9,995, 2016–2026.
- **Selection rule, fixed now:** pick the highest-CAGR config among those
  with t ≥ 3 on R per trade and max DD no worse than −20%. It goes to stage 2
  only if its CAGR also beats the baseline's development CAGR.

## Stage 2: holdout (used once, on the single selected config)
- Universe: the rotation data's non-S&P "broad" stocks, which swing_pcse has
  never used. A signal counts only if, at the signal close, price ≥ $5 and
  the 63-day median dollar volume ≥ $20M.
- **Pass** if the selected config beats the baseline on the holdout on both
  ledger CAGR and Sharpe, with t ≥ 3 and max DD no worse than −20%. Only a
  pass makes it a candidate for paper trading.

## Amendments
(none)
