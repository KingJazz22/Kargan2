"""Indicator calculations. Shared math (EMA, ATR, ADX, RSI, Stochastic, MACD,
Bollinger, Keltner, volume SMA) lives in indicators_core.py; this file adds
swing-structure detection specific to this project's strategies.
"""
import pandas as pd

from indicators_core import (  # noqa: F401
    adx,
    atr,
    bollinger_bands,
    ema,
    keltner_channel,
    macd,
    rsi,
    stochastic,
    true_range,
    volume_sma,
)


def swing_points(high: pd.Series, low: pd.Series, left: int = 5, right: int = 5):
    """Fractal swing high/low prices: bar i is a swing high if its High is the
    max over the centered window [i-left, i+right] (swing low mirrors on Low).

    Returned series are SPARSE (NaN except at the confirmation bar) and
    SHIFTED forward by `right` bars, so a pivot at bar i only appears at bar
    i+right -- the first bar it's actually knowable without lookahead.
    """
    window = left + right + 1
    centered_high_max = high.rolling(window).max().shift(-right)
    centered_low_min = low.rolling(window).min().shift(-right)

    is_high = high == centered_high_max
    is_low = low == centered_low_min

    swing_high = high.where(is_high).shift(right)
    swing_low = low.where(is_low).shift(right)
    return swing_high, swing_low


def swing_structure(high: pd.Series, low: pd.Series, left: int = 5, right: int = 5) -> pd.DataFrame:
    """Higher-high/higher-low (uptrend) vs. lower-high/lower-low (downtrend)
    swing structure, built on swing_points(). Returns a DataFrame indexed like
    `high`/`low` with:
      - resistance, support: last confirmed swing-high/low price, forward-filled
      - structure: "uptrend" (last swing high is a Higher High AND last swing
        low is a Higher Low), "downtrend" (Lower High + Lower Low), else
        "range" (mixed structure, i.e. consolidation)
    """
    swing_high, swing_low = swing_points(high, low, left, right)

    sh = swing_high.dropna()
    sl = swing_low.dropna()
    higher_high = (sh > sh.shift(1)).reindex(high.index).ffill()
    higher_low = (sl > sl.shift(1)).reindex(high.index).ffill()

    resistance = swing_high.reindex(high.index).ffill()
    support = swing_low.reindex(high.index).ffill()

    structure = pd.Series("range", index=high.index)
    structure[(higher_high == True) & (higher_low == True)] = "uptrend"  # noqa: E712
    structure[(higher_high == False) & (higher_low == False)] = "downtrend"  # noqa: E712

    return pd.DataFrame({"resistance": resistance, "support": support, "structure": structure})
