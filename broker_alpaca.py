"""Thin Alpaca wrapper for the live/paper trading runner (Breakout Hunter + MTF).

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in Kargan2/.env -- a dedicated
paper account for this project, NOT the NewBOT Alpaca account.

paper=True is hardcoded deliberately (per the decision to run paper-only).
Switching to live trading is a conscious code change, not an env-var flip.
"""
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)  # .env always wins over a stale system-level env var

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "ALPACA_API_KEY / ALPACA_SECRET_KEY not set in Kargan2/.env. "
        "Create a NEW paper-trading account at alpaca.markets dedicated to this "
        "project -- do not reuse the NewBOT account's credentials."
    )


def trading_client():
    from alpaca.trading.client import TradingClient

    return TradingClient(API_KEY, SECRET_KEY, paper=True)


def data_client():
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(API_KEY, SECRET_KEY)


def _fetch_bars(symbols: list[str], timeframe, lookback_days: int) -> dict[str, pd.DataFrame]:
    from alpaca.data.requests import StockBarsRequest

    client = data_client()
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe, start=start)
    bars = client.get_stock_bars(request).df

    out = {}
    for symbol in symbols:
        if symbol not in bars.index.get_level_values(0):
            continue
        df = bars.loc[symbol].copy()
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        out[symbol] = df[["Open", "High", "Low", "Close", "Volume"]]
    return out


def fetch_daily_bars(symbols: list[str], lookback_days: int = 400) -> dict[str, pd.DataFrame]:
    from alpaca.data.timeframe import TimeFrame

    return _fetch_bars(symbols, TimeFrame.Day, lookback_days)


def fetch_4h_bars(symbols: list[str], lookback_days: int = 400) -> dict[str, pd.DataFrame]:
    """Native 4H bars via Alpaca (no resampling needed, unlike the yfinance
    backtest path which had to build 4H from 1H due to Yahoo's interval set)."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    return _fetch_bars(symbols, TimeFrame(4, TimeFrameUnit.Hour), lookback_days)


def fetch_hourly_bars(symbols: list[str], lookback_days: int = 60) -> dict[str, pd.DataFrame]:
    from alpaca.data.timeframe import TimeFrame

    return _fetch_bars(symbols, TimeFrame.Hour, lookback_days)


def fetch_15m_bars(symbols: list[str], lookback_days: int = 15) -> dict[str, pd.DataFrame]:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    return _fetch_bars(symbols, TimeFrame(15, TimeFrameUnit.Minute), lookback_days)


def fetch_5m_bars(symbols: list[str], lookback_days: int = 10) -> dict[str, pd.DataFrame]:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    return _fetch_bars(symbols, TimeFrame(5, TimeFrameUnit.Minute), lookback_days)


def get_equity() -> float:
    return float(trading_client().get_account().equity)


def get_open_positions() -> dict[str, dict]:
    client = trading_client()
    positions = {}
    for p in client.get_all_positions():
        positions[p.symbol] = {"qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price)}
    return positions


def submit_market_buy(symbol: str, qty: float, wait_fill_seconds: int = 30) -> dict:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = trading_client()
    order = client.submit_order(MarketOrderRequest(
        symbol=symbol, qty=round(qty, 4), side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    ))

    deadline = time.time() + wait_fill_seconds
    while time.time() < deadline:
        fetched = client.get_order_by_id(order.id)
        if fetched.status == "filled":
            return {"filled": True, "fill_price": float(fetched.filled_avg_price), "qty": float(fetched.filled_qty)}
        time.sleep(2)
    return {"filled": False}


def submit_market_sell(symbol: str, qty: float, wait_fill_seconds: int = 30) -> dict:
    """Closes a long position at market -- used by PCSE's take-profit, which
    is checked periodically (has price touched the level since last check?)
    rather than resting as a second order alongside the stop-loss. Avoids
    ever having two simultaneous exit orders committing the same shares."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = trading_client()
    order = client.submit_order(MarketOrderRequest(
        symbol=symbol, qty=round(qty, 4), side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
    ))

    deadline = time.time() + wait_fill_seconds
    while time.time() < deadline:
        fetched = client.get_order_by_id(order.id)
        if fetched.status == "filled":
            return {"filled": True, "fill_price": float(fetched.filled_avg_price), "qty": float(fetched.filled_qty)}
        time.sleep(2)
    return {"filled": False}


def get_order_status(order_id: str) -> dict:
    """Used to reconcile per-strategy lots: Alpaca nets positions per symbol,
    so when two strategies hold the same symbol, checking whether a SPECIFIC
    stop order has filled is the only reliable way to know which strategy's
    lot closed."""
    order = trading_client().get_order_by_id(order_id)
    out = {"status": str(order.status)}
    if order.status == "filled":
        out["fill_price"] = float(order.filled_avg_price)
        out["fill_qty"] = float(order.filled_qty)
    return out


def submit_stop_sell(symbol: str, qty: float, stop_price: float) -> str:
    from alpaca.trading.requests import StopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = trading_client()
    order = client.submit_order(StopOrderRequest(
        symbol=symbol, qty=round(qty, 4), side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
        stop_price=round(stop_price, 2),
    ))
    return order.id


def cancel_order(order_id: str) -> None:
    trading_client().cancel_order_by_id(order_id)
