"""Renko-style brick chart: pure price-movement chart, no time axis, no candles.

Brick size is ATR-based and recalculated per brick, so it adapts to volatility
rather than staying fixed for the whole series.

Ported from OrlandMagics/renko.py -- kept in its own package (orland_ewz)
rather than Kargan2's top-level renko.py because that module is a different,
incompatible implementation (fixed brick size, different output columns)
that Kargan2's own strategy_channel_macd.py etc. depend on.
"""
import pandas as pd

from indicators_core import atr

DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULT = 1.0


def build_renko(df: pd.DataFrame, atr_period: int = DEFAULT_ATR_PERIOD,
                 atr_mult: float = DEFAULT_ATR_MULT) -> pd.DataFrame:
    """Return a DataFrame of bricks: timestamp (of formation), open, close, direction (+1/-1)."""
    atr_series = atr(df["High"], df["Low"], df["Close"], atr_period)
    closes = df["Close"]

    bricks = []
    anchor_price = None

    for i in range(len(df)):
        price = closes.iloc[i]
        ts = df.index[i]
        brick_size = atr_series.iloc[i] * atr_mult

        if anchor_price is None:
            if pd.notna(brick_size) and brick_size > 0:
                anchor_price = price
            continue

        if pd.isna(brick_size) or brick_size <= 0:
            continue

        diff = price - anchor_price
        n_bricks = int(abs(diff) // brick_size)
        if n_bricks == 0:
            continue

        direction = 1 if diff > 0 else -1
        for _ in range(n_bricks):
            brick_open = anchor_price
            brick_close = anchor_price + direction * brick_size
            bricks.append({
                "timestamp": ts,
                "open": brick_open,
                "close": brick_close,
                "direction": direction,
            })
            anchor_price = brick_close

    if not bricks:
        return pd.DataFrame(columns=["timestamp", "open", "close", "direction"])
    return pd.DataFrame(bricks)
