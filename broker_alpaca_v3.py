"""Thin Alpaca wrapper for live_trading_v3.py's dedicated paper account (the
ML flat-market long-short bot) -- separate credentials from broker_alpaca.py
(v1) and broker_alpaca_v2.py (v2), pointed at ALPACA_API_KEY_V3 /
ALPACA_SECRET_KEY_V3 in Kargan2/.env. Adds what a long-short book needs on top
of v2's interface: short sells, shortability checks, closing the whole book.

paper=True is hardcoded deliberately, same reasoning as broker_alpaca.py.
"""
import functools
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("ALPACA_API_KEY_V3")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY_V3")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "ALPACA_API_KEY_V3 / ALPACA_SECRET_KEY_V3 not set in Kargan2/.env. "
        "This is the dedicated paper account for live_trading_v3.py -- do not "
        "reuse the v1/v2 accounts."
    )

# alpaca-py's RESTClient builds a plain requests.Session() with no default
# timeout -- a stalled connection can hang every call forever. See
# broker_alpaca_v2.py's identical fix for the live incident this addresses.
REQUEST_TIMEOUT_SECONDS = 30


def _with_timeout(client):
    client._session.request = functools.partial(client._session.request, timeout=REQUEST_TIMEOUT_SECONDS)
    return client


def trading_client():
    from alpaca.trading.client import TradingClient

    return _with_timeout(TradingClient(API_KEY, SECRET_KEY, paper=True))


def data_client():
    from alpaca.data.historical import StockHistoricalDataClient

    return _with_timeout(StockHistoricalDataClient(API_KEY, SECRET_KEY))


def fetch_daily_bars(symbols: list[str], start: str) -> dict[str, pd.DataFrame]:
    """Split-adjusted daily bars (see broker_alpaca_v2._fetch_bars for why SPLIT)."""
    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                               start=pd.Timestamp(start, tz="UTC"), adjustment=Adjustment.SPLIT)
    bars = data_client().get_stock_bars(request).df
    out = {}
    for symbol in symbols:
        if symbol not in bars.index.get_level_values(0):
            continue
        df = bars.loc[symbol].copy()
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        out[symbol] = df[["Open", "High", "Low", "Close", "Volume"]]
    return out


def market_is_open() -> bool:
    return bool(trading_client().get_clock().is_open)


def get_equity() -> float:
    return float(trading_client().get_account().equity)


def get_open_positions() -> dict[str, dict]:
    positions = {}
    for p in trading_client().get_all_positions():
        positions[p.symbol] = {"qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price),
                               "market_value": float(p.market_value), "unrealized_pl": float(p.unrealized_pl)}
    return positions


def is_shortable(symbol: str) -> bool:
    a = trading_client().get_asset(symbol)
    return bool(a.tradable and a.shortable and a.easy_to_borrow)


def submit_market_order(symbol: str, qty: int, side: str, wait_fill_seconds: int = 30) -> dict:
    """side: "buy" or "sell" (a sell with no long position opens a short).
    Returns {"filled": False, "rejected": True, "error": ...} on an outright
    rejection instead of raising, same convention as broker_alpaca_v2."""
    from alpaca.common.exceptions import APIError
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    client = trading_client()
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=int(qty), side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))
    except APIError as e:
        return {"filled": False, "rejected": True, "error": str(e)}

    deadline = time.time() + wait_fill_seconds
    while time.time() < deadline:
        fetched = client.get_order_by_id(order.id)
        if fetched.status == "filled":
            return {"filled": True, "fill_price": float(fetched.filled_avg_price), "qty": float(fetched.filled_qty)}
        if fetched.status.value in ("canceled", "expired", "rejected"):
            break
        time.sleep(2)
    # Cancel on timeout instead of abandoning the order -- an abandoned DAY
    # order fills later untracked (see broker_alpaca._wait_fill_or_cancel).
    try:
        client.cancel_order_by_id(order.id)
    except APIError:
        pass  # already filled / canceled
    for _ in range(20):
        fetched = client.get_order_by_id(order.id)
        if fetched.status.value in ("filled", "canceled", "expired", "rejected", "done_for_day"):
            break
        time.sleep(1)
    filled_qty = float(fetched.filled_qty or 0)
    if filled_qty > 0:
        return {"filled": True, "fill_price": float(fetched.filled_avg_price), "qty": filled_qty}
    return {"filled": False}


def close_all_positions(wait_seconds: int = 90) -> bool:
    """Flatten the whole book (cancels open orders too). True once no positions remain."""
    client = trading_client()
    client.close_all_positions(cancel_orders=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not client.get_all_positions():
            return True
        time.sleep(3)
    return False
