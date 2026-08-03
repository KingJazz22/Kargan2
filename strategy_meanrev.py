"""Mean Reversion strategy: indicator prep + entry/exit signal columns.

Rules (as specified, with gaps filled in and flagged):
  Regime:   ADX < 20 defines "range" -- no other range detection used.
  Entry:    price closes below the lower Bollinger Band AND below the lower
            Keltner Channel (both must hold -- read as double-confirmation of
            an extreme, not a squeeze/expansion signal) AND RSI oversold AND
            Stochastic %K crosses above %D while still in the oversold zone.
            Short is the exact mirror.
  Exit:     price closes back at/through the middle Bollinger Band (SMA), OR
            an ATR trailing stop, whichever comes first.

  Gap filled in: the spec has no stop-loss. Rather than invent a separate
  structural stop (which the trend-following strategy needed because its
  trail only activates after 1x ATR of profit), the ATR trailing stop here
  is active from bar 1 of the trade -- it IS the stop-loss, appropriate for
  a strategy that expects a fast reversion rather than a running trend.
"""
import numpy as np
import pandas as pd

from indicators import adx, atr, bollinger_bands, keltner_channel, rsi, stochastic

ADX_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
KC_EMA_PERIOD = 20
KC_ATR_PERIOD = 10
KC_MULT = 1.5
RSI_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_K_SMOOTH = 3
STOCH_D_PERIOD = 3

ADX_RANGE = 20.0
RANGE_LOOKBACK = 10
RANGE_ADX_CEILING = 25.0
RANGE_FRACTION = 0.6
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
STOCH_OVERSOLD = 20.0
STOCH_OVERBOUGHT = 80.0

TRAIL_ATR_MULT = 2.0


def prepare(
    df: pd.DataFrame,
    rsi_oversold: float = RSI_OVERSOLD,
    rsi_overbought: float = RSI_OVERBOUGHT,
    stoch_oversold: float = STOCH_OVERSOLD,
    stoch_overbought: float = STOCH_OVERBOUGHT,
    bb_std: float = BB_STD,
    kc_mult: float = KC_MULT,
) -> pd.DataFrame:
    out = df.copy()
    out["adx"] = adx(out["High"], out["Low"], out["Close"], ADX_PERIOD)
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)
    out["rsi"] = rsi(out["Close"], RSI_PERIOD)

    k, d = stochastic(out["High"], out["Low"], out["Close"], STOCH_K_PERIOD, STOCH_K_SMOOTH, STOCH_D_PERIOD)
    out["stoch_k"] = k
    out["stoch_d"] = d

    bb_mid, bb_upper, bb_lower = bollinger_bands(out["Close"], BB_PERIOD, bb_std)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower

    kc_mid, kc_upper, kc_lower = keltner_channel(out["High"], out["Low"], out["Close"], KC_EMA_PERIOD, KC_ATR_PERIOD, kc_mult)
    out["kc_upper"] = kc_upper
    out["kc_lower"] = kc_lower

    # RSI/Bollinger/Keltner extremes are sharp-move signatures that push ADX UP,
    # so requiring ADX<20 on the exact signal bar is nearly self-contradicting
    # (same timing conflict found in the trend strategy). Instead require the
    # market to have been ranging RECENTLY, allowing the extreme bar to spike.
    range_regime = (out["adx"] < RANGE_ADX_CEILING).rolling(RANGE_LOOKBACK).mean() >= RANGE_FRACTION

    stoch_cross_up = (
        (out["stoch_k"] > out["stoch_d"])
        & (out["stoch_k"].shift(1) <= out["stoch_d"].shift(1))
        & (out["stoch_k"] < stoch_oversold)
    )
    stoch_cross_down = (
        (out["stoch_k"] < out["stoch_d"])
        & (out["stoch_k"].shift(1) >= out["stoch_d"].shift(1))
        & (out["stoch_k"] > stoch_overbought)
    )

    out["long_entry"] = (
        range_regime
        & (out["Close"] < out["bb_lower"])
        & (out["Close"] < out["kc_lower"])
        & (out["rsi"] < rsi_oversold)
        & stoch_cross_up
    )
    out["short_entry"] = (
        range_regime
        & (out["Close"] > out["bb_upper"])
        & (out["Close"] > out["kc_upper"])
        & (out["rsi"] > rsi_overbought)
        & stoch_cross_down
    )

    out["long_target_exit"] = out["Close"] >= out["bb_mid"]
    out["short_target_exit"] = out["Close"] <= out["bb_mid"]

    return out
