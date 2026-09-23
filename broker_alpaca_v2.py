"""Thin Alpaca wrapper for live_trading_v2.py's dedicated paper account --
completely separate credentials from broker_alpaca.py's account (which
already carries the Breakout/Swing/PCSE/Orland positions live_trading.py
manages). Identical interface to broker_alpaca.py, just pointed at
ALPACA_API_KEY_V2 / ALPACA_SECRET_KEY_V2 in Kargan2/.env.

paper=True is hardcoded deliberately, same reasoning as broker_alpaca.py.
"""
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("ALPACA_API_KEY_V2")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY_V2")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "ALPACA_API_KEY_V2 / ALPACA_SECRET_KEY_V2 not set in Kargan2/.env. "
        "This is the dedicated paper account for live_trading_v2.py -- do not "
        "reuse the ALPACA_API_KEY/ALPACA_SECRET_KEY account from broker_alpaca.py."
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
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    return _fetch_bars(symbols, TimeFrame(4, TimeFrameUnit.Hour), lookback_days)


def get_equity() -> float:
    return float(trading_client().get_account().equity)


def get_buying_power() -> float:
    return float(trading_client().get_account().buying_power)


def get_open_positions() -> dict[str, dict]:
    client = trading_client()
    positions = {}
    for p in client.get_all_positions():
        positions[p.symbol] = {"qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price)}
    return positions


def submit_market_buy(symbol: str, qty: float, wait_fill_seconds: int = 30) -> dict:
    """Returns {"filled": False, "rejected": True, "error": "..."} if Alpaca
    rejects the order outright (insufficient buying power, wash-trade
    protection, etc.) instead of letting the APIError propagate -- see
    broker_alpaca.py's identical fix for the full reasoning."""
    from alpaca.common.exceptions import APIError
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = trading_client()
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=round(qty, 4), side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        ))
    except APIError as e:
        return {"filled": False, "rejected": True, "error": str(e)}

    deadline = time.time() + wait_fill_seconds
    while time.time() < deadline:
        fetched = client.get_order_by_id(order.id)
        if fetched.status == "filled":
            return {"filled": True, "fill_price": float(fetched.filled_avg_price), "qty": float(fetched.filled_qty)}
        time.sleep(2)
    return {"filled": False}


def submit_market_sell(symbol: str, qty: float, wait_fill_seconds: int = 30) -> dict:
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
    order = trading_client().get_order_by_id(order_id)
    # order.status is a str-mixin Enum (OrderStatus.FILLED == "filled" is True),
    # but str(order.status) renders "OrderStatus.FILLED" -- silently breaking
    # every == "filled"/"canceled"/... comparison the caller does on this
    # return value. .value gives the plain "filled" string those compare against.
    out = {"status": order.status.value}
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
