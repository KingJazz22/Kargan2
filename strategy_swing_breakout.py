"""Structure-confirmed breakout: trade a breakout from a genuine consolidation
range, where resistance/support are real swing pivots (indicators.swing_structure)
rather than a naive rolling high/low. Long only for this first pass -- mirrors
strategy_breakout.py's (Breakout Hunter's) validated design, the only strategy
in this project with a confirmed positive edge (FINDINGS_BREAKOUT.md, t=3.40,
daily US equities). This variant swaps Breakout Hunter's volatility-squeeze
setup for "currently in a range/consolidation structure state" and its rolling
20-bar high for a real last-swing-high pivot as resistance.

Rules:
  Setup:   structure == "range" (mixed swing structure -- not already
           mid-trend), i.e. a genuine consolidation by swing-pivot definition.
  Entry:   close breaks above resistance (the last confirmed swing-high pivot)
           AND volume is above its own 20-bar average (same volume-confirmation
           standard used across this project, e.g. strategy_breakout.py).
  Exit:    handled by backtest_swing_breakout.py -- reuses Breakout Hunter's
           two-stage stop: a structural initial stop below the pre-breakout
           `support` swing low, then an ATR trailing stop that takes over once
           the trade has 1x ATR of profit cushion. Not reinvented here since
           that design is already validated in this codebase for breakout-style
           trades.

Gap filled in, flagged: no left/right fractal window was specified for what
counts as a "swing" pivot. Defaults to left=right=5 (a pivot needs 5 bars on
each side to confirm) -- a reasonably conservative choice for daily bars,
consistent with classic swing-trading fractal windows; not swept in this pass.
"""
import pandas as pd

from indicators import atr, swing_structure, volume_sma

ATR_PERIOD = 14
VOL_PERIOD = 20
SWING_LEFT = 5
SWING_RIGHT = 5

INITIAL_STOP_ATR_BUFFER = 0.5
TRAIL_ACTIVATE_ATR = 1.0
TRAIL_ATR_MULT = 2.5
EMERGENCY_STOP_BUFFER_ATR = 2.0  # beyond initial_stop, not independently from entry -- see FINDINGS_BREAKOUT.md


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)
    out["vol_sma"] = volume_sma(out["Volume"], VOL_PERIOD)

    structure = swing_structure(out["High"], out["Low"], SWING_LEFT, SWING_RIGHT)
    out["resistance"] = structure["resistance"]
    out["support"] = structure["support"]
    out["structure"] = structure["structure"]

    is_range = out["structure"] == "range"
    breaks_resistance = out["Close"] > out["resistance"]
    high_volume = out["Volume"] > out["vol_sma"]

    out["long_entry"] = (is_range & breaks_resistance & high_volume).fillna(False)
    return out
