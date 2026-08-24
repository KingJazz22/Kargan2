"""Multi-timeframe OHLCV data fetch via Alpaca -- the Alpaca-backed
replacement for OrlandMagics/data_fetch.py's yfinance implementation, kept
to the exact same output contract: get_multi_timeframe(symbol) returns
{timeframe: DataFrame} for keys "1D"/"1W"/"30D"/"1H"/"15m"/"5m", each with
Open/High/Low/Close/Volume columns, a tz-aware DatetimeIndex, closed bars
only.

Unlike yfinance, Alpaca has no ~60-day cap on intraday history -- 1H/15m/5m
lookbacks here are set generously for warmup (indicators like EMA(100),
regime classification) rather than against a hard provider limit. This is a
real improvement over the original OrlandMagics pipeline, though it doesn't
retroactively fix the small backtest sample that pipeline's ~60-day cap
produced -- it just means the live system isn't equally starved going
forward.

This module only ever handles the "EWZ" symbol -- see live_trading.py's
ORLAND_SYMBOL.
"""
import pandas as pd

import broker_alpaca as broker

_TIMEFRAME_MINUTES = {
    "5m": 5, "15m": 15, "1H": 60, "1D": 24 * 60, "1W": 7 * 24 * 60, "30D": 30 * 24 * 60,
}


def drop_unclosed_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop the last row if that bar's interval hasn't finished yet as of now."""
    if df is None or df.empty:
        return df
    minutes = _TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        return df

    last_ts = df.index[-1]
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo is not None else pd.Timestamp.utcnow()
    bar_close = last_ts + pd.Timedelta(minutes=minutes)
    if bar_close > now:
        return df.iloc[:-1]
    return df


def resample_ohlc(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    tz = daily.index.tz
    source = daily.tz_convert("UTC") if tz is not None else daily
    out = source.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    if tz is not None:
        out = out.tz_convert(tz)
    return out


def get_multi_timeframe(symbol: str) -> dict:
    """Return {timeframe: OHLCV DataFrame} for all 6 timeframes for `symbol`,
    each trimmed to closed candles only. `symbol` is expected to be "EWZ" --
    Alpaca's own symbol string, no ticker remapping needed (unlike WINQ26's
    ^BVSP proxy in the original yfinance pipeline, which stays alert-only and
    was never ported here)."""
    daily_map = broker.fetch_daily_bars([symbol], lookback_days=800)
    daily = daily_map.get(symbol, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
    daily = drop_unclosed_bar(daily, "1D")

    frames = {
        "1D": daily,
        "1W": drop_unclosed_bar(resample_ohlc(daily, "W"), "1W") if not daily.empty else daily,
        "30D": drop_unclosed_bar(resample_ohlc(daily, "30D"), "30D") if not daily.empty else daily,
    }

    hourly_map = broker.fetch_hourly_bars([symbol], lookback_days=60)
    frames["1H"] = drop_unclosed_bar(hourly_map.get(symbol, daily.iloc[:0]), "1H")

    m15_map = broker.fetch_15m_bars([symbol], lookback_days=15)
    frames["15m"] = drop_unclosed_bar(m15_map.get(symbol, daily.iloc[:0]), "15m")

    m5_map = broker.fetch_5m_bars([symbol], lookback_days=10)
    frames["5m"] = drop_unclosed_bar(m5_map.get(symbol, daily.iloc[:0]), "5m")

    return frames
