"""Grid/DCA martingale strategy: leg-0 entry + regime-kill signal prep.

Leg-0 entry reuses Breakout Hunter's exact squeeze+breakout signal (both
long_entry and the mirrored short_entry) unchanged -- per FINDINGS_BREAKOUT.md
it's the only entry logic in this project with a statistically significant,
regime-robust win rate (56.4-58.7%), so it's the trade-opening rule here too.
Everything past leg 0 (adding legs, sizing, exits) lives in
backtest_grid_martingale.py, not here.

regime_kill reuses strategy.py's ADX_WEAK ("dead trend") convention --
direction-agnostic, applies to an open long or short grid alike.
"""
import strategy_breakout as strat_bo
from strategy import ADX_WEAK


def prepare(df):
    out = strat_bo.prepare(df)
    out["regime_kill"] = out["adx"] < ADX_WEAK
    return out
