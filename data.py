"""Historical OHLCV data fetching. yfinance is the default (no auth needed,
covers stocks and major crypto tickers). Alpaca is optional and reads its
own credentials from this project's .env -- it is NOT the NewBOT account."""
import os

import pandas as pd
import yfinance as yf


def fetch_yfinance(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_yfinance_1h(symbol: str, period_days: int = 729) -> pd.DataFrame:
    """Native 1H bars. Yahoo caps hourly history at ~730 days regardless of symbol."""
    df = yf.download(symbol, period=f"{period_days}d", interval="60m", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No hourly data returned for {symbol} from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_yfinance_30m(symbol: str, period_days: int = 59) -> pd.DataFrame:
    """Native 30m bars. Yahoo caps this interval at ~60 days of history."""
    df = yf.download(symbol, period=f"{period_days}d", interval="30m", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No 30m data returned for {symbol} from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_yfinance_10m(symbol: str, period_days: int = 59) -> pd.DataFrame:
    """No native 10m interval on Yahoo -- built from 5m bars (also capped at
    ~60 days) via resampling, same approach as the 4H-from-1H path."""
    df = yf.download(symbol, period=f"{period_days}d", interval="5m", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No 5m data returned for {symbol} from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    resampled = df.resample("10min", origin="epoch").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    })
    return resampled.dropna()


def fetch_yfinance_4h(symbol: str, period_days: int = 729) -> pd.DataFrame:
    """Yahoo has no native 4H bars and caps hourly history at ~730 days, so we
    fetch 1H bars (the max lookback Yahoo allows) and resample to 4H ourselves."""
    df = yf.download(symbol, period=f"{period_days}d", interval="60m", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No hourly data returned for {symbol} from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    resampled = df.resample("4h", origin="epoch").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    })
    return resampled.dropna()


def fetch_yfinance_weekly(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Native 1wk bars. No Yahoo lookback cap (same as daily), unlike the
    intraday intervals -- full multi-regime history is available."""
    df = yf.download(symbol, start=start, end=end, interval="1wk", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No weekly data returned for {symbol} from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_alpaca(symbol: str, start: str, end: str, timeframe: str = "1Day") -> pd.DataFrame:
    """Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in this project's .env.
    Deliberately separate from the NewBOT Alpaca paper account credentials."""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set in Kargan2/.env. "
            "Add dedicated credentials for this project (do not reuse the NewBOT account)."
        )

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    tf = TimeFrame.Day if timeframe == "1Day" else TimeFrame.Hour
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end)
    bars = client.get_stock_bars(request).df
    bars = bars.reset_index().set_index("timestamp")
    bars = bars.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return bars[["Open", "High", "Low", "Close", "Volume"]]
