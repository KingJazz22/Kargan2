"""7-period High/Low channel fade, filtered by MACD histogram direction and
ADX<23 (ranging-regime). No intraday session-time restriction -- entries are
evaluated on every bar, whatever the bar's timeframe or construction (normal
time bars at any granularity, or Renko bricks via renko.py). The earlier
"first hour of the market" session-window gating has been removed on request
in favor of comparing bar granularity/construction instead (see
run_basket_channel_macd.py / sweep_channel_macd_bars.py).

Mean-reversion design: price overshoots a recent 7-bar high/low channel and
the bet is a snap-back to the opposite band, but only fired when ADX confirms
the market is NOT strongly trending and the MACD histogram agrees with the
reversion direction.

min_ma = 7-period SMA of Low ("min" band); max_ma = 7-period SMA of High
("max" band). Long: close < min_ma while adx<23 and macd_hist>0. Short
(mirror): close > max_ma while adx<23 and macd_hist<0. min_ma/max_ma are
ENTRY triggers only now -- exit is a fixed-ATR stop-loss/take-profit ladder
with a breakeven step, handled entirely by backtest_channel_macd.py (the
earlier "exit when price touches the opposite band" design was retired: that
moving-average target could be reached by drifting toward a losing price,
not just by price recovering -- see FINDINGS_CHANNEL_MACD.md).

At most one open position per symbol at a time is enforced in
backtest_channel_macd.py's event loop.
"""
import pandas as pd

from indicators import adx, atr, macd

CHANNEL_PERIOD = 7
ADX_PERIOD = 14
ADX_MAX = 23.0
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def prepare(
    df: pd.DataFrame,
    channel_period: int = CHANNEL_PERIOD,
    adx_period: int = ADX_PERIOD,
    adx_max: float = ADX_MAX,
    macd_fast: int = MACD_FAST,
    macd_slow: int = MACD_SLOW,
    macd_signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    out = df.copy()

    out["min_ma"] = out["Low"].rolling(channel_period).mean()
    out["max_ma"] = out["High"].rolling(channel_period).mean()
    out["adx"] = adx(out["High"], out["Low"], out["Close"], adx_period)
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)

    macd_line, signal_line, hist = macd(out["Close"], macd_fast, macd_slow, macd_signal)
    out["macd"] = macd_line
    out["macd_sig"] = signal_line
    out["macd_hist"] = hist

    adx_ok = out["adx"] < adx_max
    long_bias = out["macd_hist"] > 0
    short_bias = out["macd_hist"] < 0

    out["long_entry"] = (adx_ok & long_bias & (out["Close"] < out["min_ma"])).fillna(False)
    out["short_entry"] = (adx_ok & short_bias & (out["Close"] > out["max_ma"])).fillna(False)

    return out
