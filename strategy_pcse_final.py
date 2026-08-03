"""Probabilistic Candle-State Edge (PCSE) -- final validated configuration.

Entries: unchanged from `strategy_candle_prob.prepare()` (long_only=True) --
a walk-forward blend of a Beta-Binomial Markov model over discrete candle-
direction states and a logistic regression over candle-geometry features,
both gated on beating their fold's own base rate by a margin. Validated by a
random-entry permutation test to beat random timing at the 100th percentile
(FINDINGS_CANDLE_PROB.md).

Exit: NOT the original adaptive trailing stop from `strategy_candle_prob.py`
-- that was disproven (it destroyed the entries' edge; random entries into
that same exit engine did WORSE than the real signal, not better). Replaced,
after a systematic TP/SL/moving-average/indicator-level search
(FINDINGS_CANDLE_PROB.md, "Update: Exit Strategy Search"), with a simple
fixed two-level exit:

  - stop-loss: entry_price - SL_MULT x entry-time return-stdev, FIXED at
    entry (not trailing).
  - take-profit: price touches the BB_PERIOD-bar Bollinger upper band
    (mid + BB_NUM_STD x rolling std), checked every bar (a moving target).

This combination is deliberately patient: both a wide stop and a wide,
far-away target. That patience -- not the specific choice of Bollinger bands
over EMA/RSI/etc, several of which scored almost as well -- is the actual
finding; see FINDINGS_CANDLE_PROB.md for the full leaderboard.

Full-basket result (27 symbols, hourly, 2016-2026): n=3,821, avg_R=+0.082,
win_rate=75.5%, profit_factor=1.34, t_stat=7.11. 25/27 symbols net positive.
Caveats (exit params chosen in-sample, edge weaker in the more recent half
of history, short/other timeframes untested) are in FINDINGS_CANDLE_PROB.md
-- read them before deploying this.
"""
import strategy_candle_prob as entry_strat

SL_MULT = 12.0      # fixed stop distance, in entry-time return-stdev units
BB_PERIOD = 20
BB_NUM_STD = 3.0    # take-profit: price touches mid + BB_NUM_STD x rolling std


def prepare(df, **entry_params):
    """Pure `df -> df` convention, matching every other strategy in this
    repo: appends every column backtest_pcse_final.py needs."""
    entry_params = {"long_only": True, **entry_params}
    out = entry_strat.prepare(df, **entry_params)
    close = out["Close"]
    out["bb_mid"] = close.rolling(BB_PERIOD).mean()
    out["bb_std"] = close.rolling(BB_PERIOD).std()
    return out
