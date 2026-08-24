"""Support/resistance clustering, consolidation-zone detection, and simple
reference levels (SMA20/50/100, prior quote) -- computed per timeframe.

Feeds regime_filter (price boxed in a same-timeframe consolidation zone is
corroborating evidence for `consolidation`) and strategy_s1 (nearest
structural level for stops).

Ported from OrlandMagics/levels.py.
"""
import pandas as pd

from indicators_core import atr

from .indicators_extra import find_swings, sma

SMA_PERIODS = (20, 50, 100)
CLUSTER_TOLERANCE_ATR_MULT = 0.5
MIN_TOUCHES = 2
CONSOLIDATION_MIN_BARS_INSIDE = 10
RANGE_ATR_MULT_FOR_ZONE = 4.0  # last N bars' High-Low range <= this * ATR => compressed/consolidating


def compute_reference_levels(df: pd.DataFrame) -> dict:
    close = df["Close"]
    out = {}
    for period in SMA_PERIODS:
        out[f"sma{period}"] = float(sma(close, period).iloc[-1]) if len(close) >= period else None
    out["prior_quote"] = float(close.iloc[-2]) if len(close) >= 2 else None
    return out


def cluster_swing_levels(df: pd.DataFrame, window: int = 5,
                          tolerance_atr_mult: float = CLUSTER_TOLERANCE_ATR_MULT,
                          min_touches: int = MIN_TOUCHES) -> list:
    """Cluster nearby swing highs/lows into horizontal levels. A touch within
    `tolerance_atr_mult` * current ATR of an existing cluster joins it."""
    swings = find_swings(df, window)
    if not swings:
        return []

    atr_series = atr(df["High"], df["Low"], df["Close"])
    last_atr = atr_series.iloc[-1]
    if pd.isna(last_atr) or last_atr <= 0:
        return []
    tolerance = last_atr * tolerance_atr_mult

    clusters = []
    for _, price, kind in swings:
        for c in clusters:
            if abs(price - c["price"]) <= tolerance:
                c["prices"].append(price)
                c["price"] = sum(c["prices"]) / len(c["prices"])
                c["touches"] += 1
                c["kinds"].add(kind)
                break
        else:
            clusters.append({"price": price, "prices": [price], "touches": 1, "kinds": {kind}})

    return [c for c in clusters if c["touches"] >= min_touches]


def nearest_levels(clusters: list, current_price: float):
    supports = [c for c in clusters if c["price"] <= current_price]
    resistances = [c for c in clusters if c["price"] > current_price]
    support = max(supports, key=lambda c: c["price"]) if supports else None
    resistance = min(resistances, key=lambda c: c["price"]) if resistances else None
    return support, resistance


def detect_consolidation_zone(df: pd.DataFrame, window: int = 5,
                               tolerance_atr_mult: float = CLUSTER_TOLERANCE_ATR_MULT,
                               min_touches: int = MIN_TOUCHES,
                               min_bars_inside: int = CONSOLIDATION_MIN_BARS_INSIDE,
                               range_atr_mult: float = RANGE_ATR_MULT_FOR_ZONE):
    """Return (support_cluster, resistance_cluster, in_zone: bool).

    `in_zone` is a range-compression check: True when the High-Low range of
    the last `min_bars_inside` bars is still within `range_atr_mult` * ATR --
    i.e. price hasn't produced a move bigger than a handful of average bars'
    worth of range, which is what "boxed sideways" actually looks like.
    """
    if len(df) < min_bars_inside:
        return None, None, False

    atr_series = atr(df["High"], df["Low"], df["Close"])
    last_atr = atr_series.iloc[-1]

    in_zone = False
    if pd.notna(last_atr) and last_atr > 0:
        recent = df.tail(min_bars_inside)
        band_width = recent["High"].max() - recent["Low"].min()
        in_zone = bool(band_width <= range_atr_mult * last_atr)

    clusters = cluster_swing_levels(df, window, tolerance_atr_mult, min_touches)
    current_price = df["Close"].iloc[-1]
    support, resistance = nearest_levels(clusters, current_price) if clusters else (None, None)

    return support, resistance, in_zone


def compute_levels(df: pd.DataFrame, window: int = 5) -> dict:
    """Combine reference levels + nearest S/R + consolidation-zone flag for one timeframe."""
    ref = compute_reference_levels(df)
    support, resistance, in_zone = detect_consolidation_zone(df, window)
    return {
        **ref,
        "support": support["price"] if support else None,
        "resistance": resistance["price"] if resistance else None,
        "in_consolidation_zone": in_zone,
    }
