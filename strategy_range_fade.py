"""Range Fade: long-only daily mean-reversion, gated to flat markets.

Rules:
  Regime (flat market), one of:
    "adx"  -- ADX(14) stayed below ADX_CEILING for every one of the last
              REGIME_LOOKBACK bars (no trend has developed recently).
    "ema"  -- the EMA34 midline is flat: |EMA34[t] - EMA34[t-SLOPE_BARS]| /
              (SLOPE_BARS * ATR14) < EMA_FLAT_MAX. Same ATR-normalized slope
              FINDINGS_EMA34_REGIME.md calibrated (median ~0.08 ATR/bar).
    "both" -- both of the above.
  Trigger, one of:
    "bb"   -- close below the lower Bollinger(20, 2.0) band.
    "rsi2" -- RSI(2) below RSI2_MAX.
  Exit (decided by the caller -- backtest / live runner):
    fixed stop at signal_close - stop_atr x ATR, never trails;
    target either the Bollinger midline ("mid", recomputed daily) or a fixed
    entry + target_atr x ATR set once at entry ("fixed");
    time stop after MAX_HOLD_BARS daily bars.

Unlike strategy_meanrev.py (FINDINGS_MEANREV.md, no edge), this drops the
Keltner/Stochastic confirmations (found redundant / overfit there), gates on
a regime held over the WHOLE lookback rather than a fraction of it, and uses
a fixed ATR stop instead of a trailing one (the trailing stop fired 2-3x more
often than the mean target there).
"""
import pandas as pd

from indicators_core import adx, atr, bollinger_bands, ema, rsi

ATR_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
EMA_PERIOD = 34

REGIME_LOOKBACK = 10
ADX_CEILING = 25.0
SLOPE_BARS = 10
EMA_FLAT_MAX = 0.03
RSI2_MAX = 10.0

MAX_HOLD_BARS = 10

# Live defaults -- set from the backtest_range_fade.py result (see FINDINGS_RANGE_FADE.md).
REGIME = "both"
TRIGGER = "bb"
TARGET = "mid"
STOP_ATR = 3.0
TARGET_ATR = 1.5


def prepare(df: pd.DataFrame, regime: str = REGIME, trigger: str = TRIGGER) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)
    out["adx"] = adx(out["High"], out["Low"], out["Close"], ADX_PERIOD)
    out["rsi2"] = rsi(out["Close"], 2)
    bb_mid, bb_upper, bb_lower = bollinger_bands(out["Close"], BB_PERIOD, BB_STD)
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower

    ema34 = ema(out["Close"], EMA_PERIOD)
    out["ema_slope"] = (ema34 - ema34.shift(SLOPE_BARS)).abs() / (SLOPE_BARS * out["atr"])

    adx_quiet = (out["adx"] < ADX_CEILING).astype(float).rolling(REGIME_LOOKBACK).min() == 1.0
    ema_flat = out["ema_slope"] < EMA_FLAT_MAX
    out["regime_flat"] = {"adx": adx_quiet, "ema": ema_flat, "both": adx_quiet & ema_flat}[regime]

    if trigger == "bb":
        trig = out["Close"] < out["bb_lower"]
    elif trigger == "rsi2":
        trig = out["rsi2"] < RSI2_MAX
    elif trigger == "any":   # control: every flat-regime bar
        trig = pd.Series(True, index=out.index)
    else:
        raise ValueError(trigger)

    out["long_entry"] = (out["regime_flat"] & trig & out["atr"].notna() & out["bb_mid"].notna()).fillna(False)
    return out
