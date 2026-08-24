"""W-formation (double bottom) / M-formation (double top) detector on a Renko
brick sequence.

W resolving upward = down, up, down, up (trailing, oldest to newest) -- two
comparable-depth dips forming the two bottoms of a "W", followed by a
breakout leg. Mirrored for an M resolving downward (double top): up, down,
up, down.

Ported from OrlandMagics/pattern_w.py.
"""
import pandas as pd

from . import renko

DEFAULT_MIN_BRICKS_PER_LEG = 2
DEFAULT_LOOKBACK = 60
DEFAULT_SYMMETRY_RATIO = 2.0  # the two matching legs' brick counts must be within this ratio of each other

# Renko brick size as ATR multiples, tried from coarsest to finest -- a single
# fixed brick size (e.g. 1.0x ATR) rarely lines up with a clean W on real
# price data, so multiple granularities are checked and the first hit wins.
RENKO_ATR_MULTS = (0.50, 0.25, 0.10)


def _runs_from_end(directions: list) -> list:
    """[(direction, length), ...] for consecutive brick-direction runs, most recent first."""
    runs = []
    if not directions:
        return runs
    current = directions[-1]
    length = 0
    for d in reversed(directions):
        if d == current:
            length += 1
        else:
            runs.append((current, length))
            current = d
            length = 1
    runs.append((current, length))
    return runs


def _symmetric(a: int, b: int, ratio: float) -> bool:
    lo, hi = min(a, b), max(a, b)
    return hi <= lo * ratio


def find_w_formation(bricks: pd.DataFrame, min_bricks_per_leg: int = DEFAULT_MIN_BRICKS_PER_LEG,
                      lookback: int = DEFAULT_LOOKBACK,
                      symmetry_ratio: float = DEFAULT_SYMMETRY_RATIO) -> str | None:
    """Return 'up' for a completed W (double bottom), 'down' for a completed
    M (double top), else None. Checks the 4 most recent brick-direction runs."""
    if bricks is None or len(bricks) < 4 * min_bricks_per_leg:
        return None

    directions = bricks["direction"].tail(lookback).tolist()
    runs = _runs_from_end(directions)
    if len(runs) < 4:
        return None

    r0, r1, r2, r3 = runs[0], runs[1], runs[2], runs[3]  # r0 = most recent
    if any(r[1] < min_bricks_per_leg for r in (r0, r1, r2, r3)):
        return None

    # oldest -> newest: r3, r2, r1, r0
    if r3[0] == -1 and r2[0] == 1 and r1[0] == -1 and r0[0] == 1:
        if _symmetric(r3[1], r1[1], symmetry_ratio):
            return "up"
    if r3[0] == 1 and r2[0] == -1 and r1[0] == 1 and r0[0] == -1:
        if _symmetric(r3[1], r1[1], symmetry_ratio):
            return "down"
    return None


def find_w_formation_multi(df: pd.DataFrame, atr_mults=RENKO_ATR_MULTS,
                            min_bricks_per_leg: int = DEFAULT_MIN_BRICKS_PER_LEG,
                            lookback: int = DEFAULT_LOOKBACK):
    """Try Renko brick sizes from coarsest to finest (ATR multiples). Returns
    (direction, completion_timestamp, atr_mult_used) for the first size that
    finds a resolved W/M, else (None, None, None)."""
    for mult in atr_mults:
        bricks = renko.build_renko(df, atr_mult=mult)
        direction = find_w_formation(bricks, min_bricks_per_leg=min_bricks_per_leg, lookback=lookback)
        if direction is not None:
            ts = bricks["timestamp"].iloc[-1] if not bricks.empty else None
            return direction, ts, mult
    return None, None, None
