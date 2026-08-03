"""Pure OHLCV-derived candle features for the Probabilistic Candle-State Edge
(PCSE) strategy: candle geometry ratios and self-referential (rolling, not
fixed-magic-number) statistics only -- deliberately no canned indicators
(RSI/MACD/ADX/Bollinger/etc), per the strategy's brief to open trades from
candle observation + statistics alone.

Every rolling window here is trailing-inclusive of the bar being scored (it
may use that bar's own OHLC, plus strictly prior bars for the window stats)
-- never a future bar. That's what "no lookahead" means throughout this repo.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "clv",
    "return_zscore", "range_zscore", "range_pctile", "gap_zscore", "volume_zscore",
]


def body_ratio(df: pd.DataFrame) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    return ((df["Close"] - df["Open"]) / rng).fillna(0.0)


def upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    return ((df["High"] - df[["Open", "Close"]].max(axis=1)) / rng).fillna(0.0)


def lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    return ((df[["Open", "Close"]].min(axis=1) - df["Low"]) / rng).fillna(0.0)


def close_location_value(df: pd.DataFrame) -> pd.Series:
    """CLV: +1 = closed at the high, -1 = closed at the low, 0 = closed mid-range."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    return clv.fillna(0.0)


def log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def range_pct(df: pd.DataFrame) -> pd.Series:
    return (df["High"] - df["Low"]) / df["Close"]


def gap_return(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return (df["Open"] - prev_close) / prev_close


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank (0-1) of the trailing window's last value, inclusive."""
    return series.rolling(window, min_periods=window).rank(pct=True)


def candle_direction(df: pd.DataFrame, flat_eps: float = 1e-4) -> pd.Series:
    ret = (df["Close"] - df["Open"]) / df["Open"]
    return pd.Series(
        np.where(ret > flat_eps, "U", np.where(ret < -flat_eps, "D", "F")),
        index=df.index,
    )


def direction_state(df: pd.DataFrame, k: int = 3, flat_eps: float = 1e-4) -> pd.Series:
    """Last-k-candle direction sequence as a discrete state string, e.g. 'UUD'
    (oldest -> newest, left -> right). NaN for the first k-1 bars."""
    direction = candle_direction(df, flat_eps)
    parts = [direction.shift(k - 1 - i) for i in range(k)]
    state = parts[0].str.cat(parts[1:])
    state.index = df.index
    return state


def build_features(df: pd.DataFrame, *, zscore_window: int = 100, vol_window: int = 20,
                    k_state: int = 3, flat_eps: float = 1e-4) -> pd.DataFrame:
    """Append every candle feature + the discrete direction-state to a copy of df."""
    out = df.copy()
    out["body_ratio"] = body_ratio(out)
    out["upper_wick_ratio"] = upper_wick_ratio(out)
    out["lower_wick_ratio"] = lower_wick_ratio(out)
    out["clv"] = close_location_value(out)
    out["log_return"] = log_return(out["Close"])
    out["return_zscore"] = rolling_zscore(out["log_return"], zscore_window)
    out["range_pct"] = range_pct(out)
    out["range_zscore"] = rolling_zscore(out["range_pct"], zscore_window)
    out["range_pctile"] = rolling_percentile_rank(out["range_pct"], zscore_window)
    out["gap_return"] = gap_return(out)
    out["gap_zscore"] = rolling_zscore(out["gap_return"], zscore_window)
    if "Volume" in out.columns and out["Volume"].abs().sum() > 0:
        out["volume_zscore"] = rolling_zscore(out["Volume"], vol_window)
    else:
        out["volume_zscore"] = 0.0
    out["direction_state"] = direction_state(out, k=k_state, flat_eps=flat_eps)
    # rolling stdev of returns, used by the strategy for volatility-normalized
    # stops -- a raw statistical measure, not the ATR indicator.
    out["ret_stdev"] = out["log_return"].rolling(zscore_window, min_periods=zscore_window).std(ddof=0)
    return out
