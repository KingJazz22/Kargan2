"""Indicator functions ported from OrlandMagics/indicators.py that aren't in
Kargan2's own top-level indicators.py. Shared basics (ema, atr, adx, etc.)
come from Kargan2's top-level indicators_core.py -- imported here rather than
duplicated again, so this package stays in sync with the rest of Kargan2's
indicator math automatically.
"""
import pandas as pd

from indicators_core import ema  # noqa: F401


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rolling_percentile_rank(series: pd.Series, lookback: int = 100) -> pd.Series:
    """Percentile rank (0-1) of each value within its own trailing `lookback` window."""
    return series.rolling(lookback).apply(
        lambda w: (w <= w[-1]).sum() / len(w), raw=True
    )


def force_index(close: pd.Series, volume: pd.Series, period: int = 2) -> pd.Series:
    """Elder's Force Index: (close - prior close) * volume, smoothed with an EMA."""
    raw = close.diff() * volume
    return ema(raw, period)


def find_swings(df: pd.DataFrame, window: int = 5):
    """Local-extrema swing highs/lows. Returns a list of (position, price, 'high'|'low')
    sorted by position. A bar is a swing high/low if it's the max/min of the
    `window`-bar window centered on it."""
    highs = df["High"]
    lows = df["Low"]
    swings = []
    n = len(df)
    for i in range(window, n - window):
        window_highs = highs.iloc[i - window: i + window + 1]
        if highs.iloc[i] == window_highs.max():
            swings.append((i, float(highs.iloc[i]), "high"))
        window_lows = lows.iloc[i - window: i + window + 1]
        if lows.iloc[i] == window_lows.min():
            swings.append((i, float(lows.iloc[i]), "low"))
    swings.sort(key=lambda s: s[0])
    return swings
