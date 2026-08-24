"""Multi-Timeframe RSI + Stochastic strategy.

Rules (as specified, with gaps filled in and flagged):
  Entry:  RSI AND Stochastic %K both oversold, simultaneously, on BOTH the
          execution timeframe (LTF=1H) and a confirming higher timeframe
          (HTF=4H). Short is the exact mirror (both overbought, both TFs).
  Exit:   "wait for price to move back" is read as: the LTF RSI recovers
          back through the neutral 50 level (the oversold/overbought
          condition resolving) -- close for whatever profit/loss exists then.

  Gap filled in: no stop-loss was specified (same gap as the mean-reversion
  spec). Reusing the same fix: an ATR trailing stop, active from bar 1 of
  the trade, doubles as the protective stop.

  HTF alignment (lookahead safety): a resampled HTF bar labeled with
  timestamp T covers [T, T+htf_bar_hours) and its OHLC/indicators aren't
  fully known until T+htf_bar_hours. The HTF frame's index is shifted
  forward by that amount before an as-of backward join onto the LTF index,
  so an LTF bar only ever sees an HTF reading that has actually closed.
"""
import pandas as pd

from indicators import atr, rsi, stochastic

RSI_PERIOD = 14
ATR_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_K_SMOOTH = 3
STOCH_D_PERIOD = 3

RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
STOCH_OVERSOLD = 20.0
STOCH_OVERBOUGHT = 80.0
RSI_EXIT_NEUTRAL = 50.0

TRAIL_ATR_MULT = 2.0


def prepare(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    htf_bar_hours: float = 4,
    rsi_oversold: float = RSI_OVERSOLD,
    rsi_overbought: float = RSI_OVERBOUGHT,
    stoch_oversold: float = STOCH_OVERSOLD,
    stoch_overbought: float = STOCH_OVERBOUGHT,
    rsi_exit_neutral: float = RSI_EXIT_NEUTRAL,
    long_only: bool = False,
    trend_filter_df: pd.DataFrame = None,
    trend_filter_period: int = 200,
    trend_filter_bar_hours: float = 24,
    trend_filter_type: str = "sma",
    tsl_only_exit: bool = False,
) -> pd.DataFrame:
    out = ltf_df.copy()
    out["rsi"] = rsi(out["Close"], RSI_PERIOD)
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)
    k, d = stochastic(out["High"], out["Low"], out["Close"], STOCH_K_PERIOD, STOCH_K_SMOOTH, STOCH_D_PERIOD)
    out["stoch_k"] = k

    htf = htf_df.copy()
    htf["htf_rsi"] = rsi(htf["Close"], RSI_PERIOD)
    htf_k, htf_d = stochastic(htf["High"], htf["Low"], htf["Close"], STOCH_K_PERIOD, STOCH_K_SMOOTH, STOCH_D_PERIOD)
    htf["htf_stoch_k"] = htf_k
    htf_known = htf[["htf_rsi", "htf_stoch_k"]].copy()
    htf_known.index = htf_known.index + pd.Timedelta(hours=htf_bar_hours)
    if out.index.tz is not None and htf_known.index.tz is None:
        htf_known.index = htf_known.index.tz_localize(out.index.tz)
    elif out.index.tz is None and htf_known.index.tz is not None:
        out.index = out.index.tz_localize(htf_known.index.tz)

    # pd.Timedelta arithmetic can upcast the index's datetime64 unit (e.g.
    # [s] -> [us]), which makes merge_asof reject the two indexes as
    # "incompatible merge keys" even though they're both plain UTC
    # timestamps -- normalize both sides to the same unit first.
    if out.index.dtype != htf_known.index.dtype:
        common_unit = "us"
        out.index = out.index.as_unit(common_unit)
        htf_known.index = htf_known.index.as_unit(common_unit)

    out = pd.merge_asof(out.sort_index(), htf_known.sort_index(), left_index=True, right_index=True, direction="backward")

    ltf_oversold = (out["rsi"] < rsi_oversold) & (out["stoch_k"] < stoch_oversold)
    ltf_overbought = (out["rsi"] > rsi_overbought) & (out["stoch_k"] > stoch_overbought)
    htf_oversold = (out["htf_rsi"] < rsi_oversold) & (out["htf_stoch_k"] < stoch_oversold)
    htf_overbought = (out["htf_rsi"] > rsi_overbought) & (out["htf_stoch_k"] > stoch_overbought)

    long_entry = ltf_oversold & htf_oversold
    short_entry = ltf_overbought & htf_overbought

    if trend_filter_df is not None:
        # Daily trend line (SMA/EMA/VWAP, default 200-period SMA) as a filter:
        # only take longs above it, shorts below it. Same lookahead-safe
        # shifted-asof join as the HTF alignment above -- the daily bar isn't
        # known until it closes.
        tf = trend_filter_df.copy()
        if trend_filter_type == "sma":
            tf["trend_line"] = tf["Close"].rolling(trend_filter_period).mean()
        elif trend_filter_type == "ema":
            tf["trend_line"] = tf["Close"].ewm(span=trend_filter_period, adjust=False).mean()
        elif trend_filter_type == "vwap":
            typical = (tf["High"] + tf["Low"] + tf["Close"]) / 3
            cum_tpv = (typical * tf["Volume"]).rolling(trend_filter_period).sum()
            cum_vol = tf["Volume"].rolling(trend_filter_period).sum()
            tf["trend_line"] = cum_tpv / cum_vol
        else:
            raise ValueError(f"unknown trend_filter_type: {trend_filter_type!r}")
        tf_known = tf[["trend_line"]].copy()
        tf_known.index = tf_known.index + pd.Timedelta(hours=trend_filter_bar_hours)
        if out.index.tz is not None and tf_known.index.tz is None:
            tf_known.index = tf_known.index.tz_localize(out.index.tz)
        if out.index.dtype != tf_known.index.dtype:
            common_unit = "us"
            out.index = out.index.as_unit(common_unit)
            tf_known.index = tf_known.index.as_unit(common_unit)
        out = pd.merge_asof(out.sort_index(), tf_known.sort_index(), left_index=True, right_index=True, direction="backward")

        above_trend = out["Close"] > out["trend_line"]
        long_entry = long_entry & above_trend
        short_entry = short_entry & ~above_trend

    out["long_entry"] = long_entry.fillna(False)
    out["short_entry"] = short_entry.fillna(False) if not long_only else False

    if tsl_only_exit:
        # v2: no RSI-neutral profit target -- the ATR trailing stop (already
        # active from bar 1, see module docstring) is the ONLY exit, so
        # winners aren't cut off at the first neutral-RSI touch.
        out["long_target_exit"] = False
        out["short_target_exit"] = False
    else:
        out["long_target_exit"] = out["rsi"] >= rsi_exit_neutral
        out["short_target_exit"] = out["rsi"] <= rsi_exit_neutral

    return out
