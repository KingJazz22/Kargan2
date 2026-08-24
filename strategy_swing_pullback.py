"""Trend-continuation pullback: in an established uptrend swing structure
(last swing high is a Higher High AND last swing low is a Higher Low, per
indicators.swing_structure), buy a pullback to the current support level that
gets defended (reclaimed) rather than broken. A genuinely different edge shape
from strategy_swing_breakout.py -- this is a continuation entry inside an
already-trending market, not a breakout from a range, and it is NOT a fade of
a range: FINDINGS_CHANNEL_MACD.md found range-fading to be a severe, decisive
loser in this codebase at every configuration tried. Long only for this first
pass.

Rules:
  Setup:   structure == "uptrend".
  Entry:   the bar's Low comes within PULLBACK_ATR_MULT x ATR of the current
           support level (a genuine pullback touch, not just "price is above
           support") AND the bar's Close is back above support (support held
           / was reclaimed, not broken).
  Exit:    handled by backtest_swing_pullback.py -- same two-stage stop shape
           reused from strategy_swing_breakout.py / Breakout Hunter: a
           structural initial stop just below the defended support level,
           then an ATR trailing stop once the trade has 1x ATR of profit
           cushion.

Gaps filled in, flagged:
- Fractal window for swing pivots: same left=right=5 default as
  strategy_swing_breakout.py, for consistency between the two variants.
- "Pulls back to support" needed a tolerance band since price rarely touches
  a level exactly -- PULLBACK_ATR_MULT=1.0 (within 1x ATR of support) is an
  untuned starting point, not swept in this pass.
"""
import pandas as pd

from indicators import atr, swing_structure

ATR_PERIOD = 14
SWING_LEFT = 5
SWING_RIGHT = 5
PULLBACK_ATR_MULT = 1.0

INITIAL_STOP_ATR_BUFFER = 0.5
TRAIL_ACTIVATE_ATR = 1.0
TRAIL_ATR_MULT = 2.5
EMERGENCY_STOP_BUFFER_ATR = 2.0  # beyond initial_stop, not independently from entry -- see FINDINGS_BREAKOUT.md


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)

    structure = swing_structure(out["High"], out["Low"], SWING_LEFT, SWING_RIGHT)
    out["resistance"] = structure["resistance"]
    out["support"] = structure["support"]
    out["structure"] = structure["structure"]

    in_uptrend = out["structure"] == "uptrend"
    near_support = out["Low"] <= (out["support"] + PULLBACK_ATR_MULT * out["atr"])
    reclaim = out["Close"] > out["support"]

    out["long_entry"] = (in_uptrend & near_support & reclaim).fillna(False)
    return out
