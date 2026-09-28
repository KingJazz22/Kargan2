"""Alpaca-backed OHLCV fetching for deep backtests. Alpaca's free/IEX data
gives ~10 years of hourly/4H equity bars (since 2016) and ~5.6 years of
hourly crypto bars (since 2021) -- vastly deeper than yfinance's ~2yr
hourly/4H cap, which was the binding constraint on every MTF/crypto/4H
backtest window earlier in this project. Uses Kargan2's own dedicated
Alpaca credentials (Kargan2/.env), same account live_trading.py trades on.
"""
import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)

from alpaca.data.enums import Adjustment
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

_stock_client = None
_crypto_client = None


def stock_client():
    global _stock_client
    if _stock_client is None:
        _stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    return _stock_client


def crypto_client():
    global _crypto_client
    if _crypto_client is None:
        _crypto_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
    return _crypto_client


def _bars_to_df(bars_df, symbol):
    if symbol not in bars_df.index.get_level_values(0):
        raise ValueError(f"no bars for {symbol}")
    df = bars_df.loc[symbol].copy()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_stock(symbol, timeframe, start="2016-01-01", end="2026-08-01", retries=3):
    # Alpaca defaults to RAW (unadjusted) bars -- a stock split shows up as a
    # fake single-day price collapse (e.g. WMT's Feb 2024 3-for-1 split reads
    # as a -66% one-day return), corrupting any return/volatility statistic
    # computed off Close. ALL matches data.py's yfinance fetchers, which use
    # auto_adjust=True (split + dividend adjusted) by default.
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end,
                            adjustment=Adjustment.ALL)
    for attempt in range(retries):
        try:
            bars = stock_client().get_stock_bars(req).df
            return _bars_to_df(bars, symbol)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_crypto(symbol, timeframe, start="2021-01-01", end="2026-08-01", retries=3):
    req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
    for attempt in range(retries):
        try:
            bars = crypto_client().get_crypto_bars(req).df
            return _bars_to_df(bars, symbol)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_stock_daily(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame.Day, start, end)


def fetch_stock_4h(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame(4, TimeFrameUnit.Hour), start, end)


def fetch_stock_1h(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame(1, TimeFrameUnit.Hour), start, end)


def fetch_stock_30m(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame(30, TimeFrameUnit.Minute), start, end)


def fetch_stock_15m(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame(15, TimeFrameUnit.Minute), start, end)


def fetch_stock_10m(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame(10, TimeFrameUnit.Minute), start, end)


def fetch_stock_5m(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame(5, TimeFrameUnit.Minute), start, end)


def fetch_stock_2m(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame(2, TimeFrameUnit.Minute), start, end)


def fetch_stock_1m(symbol, start="2016-01-01", end="2026-08-01"):
    return fetch_stock(symbol, TimeFrame(1, TimeFrameUnit.Minute), start, end)


def fetch_crypto_daily(symbol, start="2021-01-01", end="2026-08-01"):
    return fetch_crypto(symbol, TimeFrame.Day, start, end)


def fetch_crypto_4h(symbol, start="2021-01-01", end="2026-08-01"):
    return fetch_crypto(symbol, TimeFrame(4, TimeFrameUnit.Hour), start, end)


def fetch_crypto_1h(symbol, start="2021-01-01", end="2026-08-01"):
    return fetch_crypto(symbol, TimeFrame(1, TimeFrameUnit.Hour), start, end)


def fetch_crypto_30m(symbol, start="2021-01-01", end="2026-08-01"):
    return fetch_crypto(symbol, TimeFrame(30, TimeFrameUnit.Minute), start, end)


def fetch_crypto_15m(symbol, start="2021-01-01", end="2026-08-01"):
    return fetch_crypto(symbol, TimeFrame(15, TimeFrameUnit.Minute), start, end)


def fetch_crypto_10m(symbol, start="2021-01-01", end="2026-08-01"):
    return fetch_crypto(symbol, TimeFrame(10, TimeFrameUnit.Minute), start, end)


def fetch_crypto_5m(symbol, start="2021-01-01", end="2026-08-01"):
    return fetch_crypto(symbol, TimeFrame(5, TimeFrameUnit.Minute), start, end)


def fetch_crypto_1m(symbol, start="2021-01-01", end="2026-08-01"):
    return fetch_crypto(symbol, TimeFrame(1, TimeFrameUnit.Minute), start, end)
