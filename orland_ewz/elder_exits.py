"""Exit rule for the Elder Triple Screen entries (elder_strategy.py).

Ported from OrlandMagics/elder_exits.py -- that file has 15 exit-rule
families (most of them barely tested variants used only for the exit-rule
comparison sweep). Only `make_adx_fade` is ported here: it's the one
OrlandMagics' own cross-asset report found to generalize across both EWZ and
WINQ26 (at threshold=25, not the EWZ-only-optimized threshold=20), and it's
the only one this project decided to actually wire into live trading. Not
porting the other 14 keeps this file from carrying 400+ lines of unvalidated,
unused exit logic into live-trading code.

Common signature: exit_rule(position, bar, context) -> Optional[str] reason.
`context` carries:
  atr_val    -- current ATR(14)
  df         -- OHLCV history truncated through the current bar
  bars_held  -- number of bars since entry (inclusive of the entry bar)
Rules may mutate position.stop / position.trail_active / position.extreme_price.
"""
import pandas as pd

from indicators_core import adx


def _hard_stop_hit(position, bar) -> bool:
    if position.direction == "up":
        return bar["Low"] <= position.stop
    return bar["High"] >= position.stop


def make_adx_fade(threshold: float):
    """Exit once ADX(14) drops below `threshold` -- the trend that justified
    entry is fading."""
    def _rule(position, bar, context):
        if _hard_stop_hit(position, bar):
            return "stop"
        df = context["df"]
        adx_val = adx(df["High"], df["Low"], df["Close"]).iloc[-1]
        if pd.notna(adx_val) and adx_val < threshold:
            return "adx_fade"
        return None
    return _rule
