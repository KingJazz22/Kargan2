"""3-Timeframe RSI + Stochastic alignment strategy.

Extends strategy_mtf.py's 2-timeframe (LTF+HTF) alignment to 3: RSI(14) AND
Stochastic(14,3,3) %K must both be oversold (or overbought) simultaneously
on the execution timeframe (LTF) AND a middle confirming timeframe (MTF) AND
a higher confirming timeframe (HTF) -- i.e. "3 of {1m..4h} align" gates the
entry. Short is the exact mirror. Same engine as strategy_mtf.py otherwise:
exit is either the RSI-neutral-50 recovery (v1) or TSL-only (v2, default
here per FINDINGS_MTF.md, which found TSL-only strictly beats the RSI-neutral
exit on every timeframe pairing tested), and the same ATR trailing stop
(active from bar 1, doubling as the stop-loss) drives both position sizing
and the exit -- unchanged from strategy_mtf.py.

Lookahead safety: each higher timeframe (MTF, then HTF) is joined onto the
LTF index via the same shifted-index `merge_asof` pattern strategy_mtf.py
already uses twice (once for its HTF, once for its optional trend filter) --
a bar's own indicators aren't visible to the LTF until that bar has actually
closed (index shifted forward by the bar's own duration before an as-of
backward join).
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


def _add_rsi_stoch(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}rsi"] = rsi(out["Close"], RSI_PERIOD)
    k, _d = stochastic(out["High"], out["Low"], out["Close"], STOCH_K_PERIOD, STOCH_K_SMOOTH, STOCH_D_PERIOD)
    out[f"{prefix}stoch_k"] = k
    return out


def _join_higher_tf(ltf: pd.DataFrame, htf: pd.DataFrame, prefix: str, bar_minutes: float) -> pd.DataFrame:
    known = htf[[f"{prefix}rsi", f"{prefix}stoch_k"]].copy()
    known.index = known.index + pd.Timedelta(minutes=bar_minutes)
    if ltf.index.tz is not None and known.index.tz is None:
        known.index = known.index.tz_localize(ltf.index.tz)
    elif ltf.index.tz is None and known.index.tz is not None:
        ltf = ltf.copy()
        ltf.index = ltf.index.tz_localize(known.index.tz)
    # pd.Timedelta arithmetic can upcast the index's datetime64 unit (e.g.
    # [s] -> [us]), which makes merge_asof reject the two indexes as
    # "incompatible merge keys" even though they're both plain UTC
    # timestamps -- normalize both sides to the same unit first.
    if ltf.index.dtype != known.index.dtype:
        common_unit = "us"
        ltf = ltf.copy()
        ltf.index = ltf.index.as_unit(common_unit)
        known.index = known.index.as_unit(common_unit)
    return pd.merge_asof(ltf.sort_index(), known.sort_index(), left_index=True, right_index=True, direction="backward")


def prepare(
    ltf_df: pd.DataFrame,
    mtf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    mtf_bar_minutes: float,
    htf_bar_minutes: float,
    rsi_oversold: float = RSI_OVERSOLD,
    rsi_overbought: float = RSI_OVERBOUGHT,
    stoch_oversold: float = STOCH_OVERSOLD,
    stoch_overbought: float = STOCH_OVERBOUGHT,
    rsi_exit_neutral: float = RSI_EXIT_NEUTRAL,
    long_only: bool = True,
    tsl_only_exit: bool = True,
) -> pd.DataFrame:
    out = _add_rsi_stoch(ltf_df, "")
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)

    mtf = _add_rsi_stoch(mtf_df, "mtf_")
    htf = _add_rsi_stoch(htf_df, "htf_")

    out = _join_higher_tf(out, mtf, "mtf_", mtf_bar_minutes)
    out = _join_higher_tf(out, htf, "htf_", htf_bar_minutes)

    ltf_oversold = (out["rsi"] < rsi_oversold) & (out["stoch_k"] < stoch_oversold)
    ltf_overbought = (out["rsi"] > rsi_overbought) & (out["stoch_k"] > stoch_overbought)
    mtf_oversold = (out["mtf_rsi"] < rsi_oversold) & (out["mtf_stoch_k"] < stoch_oversold)
    mtf_overbought = (out["mtf_rsi"] > rsi_overbought) & (out["mtf_stoch_k"] > stoch_overbought)
    htf_oversold = (out["htf_rsi"] < rsi_oversold) & (out["htf_stoch_k"] < stoch_oversold)
    htf_overbought = (out["htf_rsi"] > rsi_overbought) & (out["htf_stoch_k"] > stoch_overbought)

    long_entry = ltf_oversold & mtf_oversold & htf_oversold
    short_entry = ltf_overbought & mtf_overbought & htf_overbought

    out["long_entry"] = long_entry.fillna(False)
    out["short_entry"] = short_entry.fillna(False) if not long_only else False

    if tsl_only_exit:
        out["long_target_exit"] = False
        out["short_target_exit"] = False
    else:
        out["long_target_exit"] = out["rsi"] >= rsi_exit_neutral
        out["short_target_exit"] = out["rsi"] <= rsi_exit_neutral

    return out
