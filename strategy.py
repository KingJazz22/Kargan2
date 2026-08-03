"""Adaptive Trend Following strategy: indicator prep + entry/exit signal columns.

Rules (as refined in discussion):
  Regime:   EMA50 vs EMA200 sets direction; ADX tiers gate entries/exit sensitivity.
              ADX >= 25            -> full trend, entries allowed
              20 <= ADX < 25       -> weakening, no new entries, tighter exit
              ADX < 20             -> dead trend, hard exit any open position
  Entry:    price touches EMA50 (low<=EMA50<=high for long) after having closed
            above EMA50 for the prior CONFIRM_BARS bars, closed with a bullish
            reclaim candle, and volume above its 20-bar average. Short is the mirror.
  Stops:    initial stop = pullback swing low/high minus/plus 0.5*ATR (structural).
            once unrealized profit >= 1*ATR, switch to a trailing stop at
            2.5*ATR off the highest high / lowest low since entry.
            emergency backstop = 4*ATR from entry (should rarely bind).
  Exits:    close back beyond EMA50 against position direction (fast reversal),
            EMA50/EMA200 cross against position (slow backstop),
            ADX < 20 (hard kill).
"""
import numpy as np
import pandas as pd

from indicators import adx, atr, ema, volume_sma

EMA_FAST = 50
EMA_SLOW = 200
ADX_PERIOD = 14
ATR_PERIOD = 14
VOL_PERIOD = 20
CONFIRM_BARS = 3
ADX_LOOKBACK = 10

ADX_TREND = 25.0
ADX_WEAK = 20.0

INITIAL_STOP_ATR_BUFFER = 0.5
TRAIL_ACTIVATE_ATR = 1.0
TRAIL_ATR_MULT = 2.5
EMERGENCY_ATR_MULT = 4.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator + signal columns to an OHLCV dataframe (High/Low/Open/Close/Volume)."""
    out = df.copy()
    out["ema_fast"] = ema(out["Close"], EMA_FAST)
    out["ema_slow"] = ema(out["Close"], EMA_SLOW)
    out["adx"] = adx(out["High"], out["Low"], out["Close"], ADX_PERIOD)
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)
    out["vol_sma"] = volume_sma(out["Volume"], VOL_PERIOD)

    uptrend = out["ema_fast"] > out["ema_slow"]
    downtrend = out["ema_fast"] < out["ema_slow"]

    was_above = out["Close"].shift(1) > out["ema_fast"].shift(1)
    confirmed_above = was_above.rolling(CONFIRM_BARS).apply(lambda x: bool(np.all(x)), raw=True).astype(bool)
    was_below = out["Close"].shift(1) < out["ema_fast"].shift(1)
    confirmed_below = was_below.rolling(CONFIRM_BARS).apply(lambda x: bool(np.all(x)), raw=True).astype(bool)

    touches_from_above = (out["Low"] <= out["ema_fast"]) & (out["High"] >= out["ema_fast"])
    touches_from_below = touches_from_above  # same geometric condition, direction comes from trend/candle

    body_range = (out["High"] - out["Low"]).replace(0, np.nan)
    bullish_candle = (
        (out["Close"] > out["Open"])
        & (((out["Close"] - out["Low"]) / body_range) > 0.5)
    )
    bearish_candle = (
        (out["Close"] < out["Open"])
        & (((out["High"] - out["Close"]) / body_range) > 0.5)
    )

    vol_confirm = out["Volume"] > out["vol_sma"]

    # ADX cools off during the pullback itself (countertrend move balances +DM/-DM),
    # so require the trend to have been strong RECENTLY rather than on the touch bar.
    adx_was_strong = (out["adx"] >= ADX_TREND).rolling(ADX_LOOKBACK).max().astype(bool)

    out["long_entry"] = (
        uptrend
        & adx_was_strong
        & confirmed_above
        & touches_from_above
        & bullish_candle
        & vol_confirm
    )
    out["short_entry"] = (
        downtrend
        & adx_was_strong
        & confirmed_below
        & touches_from_below
        & bearish_candle
        & vol_confirm
    )

    out["long_reversal_exit"] = out["Close"] < out["ema_fast"]
    out["short_reversal_exit"] = out["Close"] > out["ema_fast"]
    out["long_backstop_exit"] = out["ema_fast"] < out["ema_slow"]
    out["short_backstop_exit"] = out["ema_fast"] > out["ema_slow"]
    out["hard_kill"] = out["adx"] < ADX_WEAK

    return out
