"""Thin Alpaca wrapper for the live/paper trading runner (Breakout Hunter + MTF).

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in Kargan2/.env -- a dedicated
paper account for this project, NOT the NewBOT Alpaca account.

paper=True is hardcoded deliberately (per the decision to run paper-only).
Switching to live trading is a conscious code change, not an env-var flip.
"""
import functools
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

# alpaca-py's RESTClient builds a plain requests.Session() with no default
# timeout (confirmed in alpaca.common.rest.RESTClient.__init__/_one_request) --
# a stalled connection can hang every call forever, which live_trading.py's
# main() can't recover from: reconcile() runs OUTSIDE main()'s try/except, so
# a hang there freezes railway_runner.py's entire scheduler loop, silently,
# with no error logged (seen live 2026-10-28, same bug, on the v2 account).
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


def get_buying_power() -> float:
    return float(trading_client().get_account().buying_power)


def get_open_positions() -> dict[str, dict]:
    client = trading_client()
    positions = {}
    for p in client.get_all_positions():
        positions[p.symbol] = {"qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price),
                                "current_price": float(p.current_price)}
    return positions


def is_market_open() -> bool:
    return bool(trading_client().get_clock().is_open)


TERMINAL_STATUSES = ("filled", "canceled", "expired", "rejected", "done_for_day")


def _cancel_and_collect(client, order_id, timeout_seconds: int = 20):
    """Cancels an order and waits until Alpaca confirms a terminal status.
    cancel_order_by_id only REQUESTS the cancel -- the shares it reserved stay
    held (and the order can still fill) until the status actually flips, so
    anything submitted against the same shares right after a bare cancel gets
    "insufficient qty available" (how HD's v2 take-profit aborted and left the
    position with no stop). Returns the final order object."""
    from alpaca.common.exceptions import APIError

    try:
        client.cancel_order_by_id(order_id)
    except APIError:
        pass  # already filled / canceled -- the status below says which
    deadline = time.time() + timeout_seconds
    while True:
        fetched = client.get_order_by_id(order_id)
        if fetched.status.value in TERMINAL_STATUSES or time.time() >= deadline:
            return fetched
        time.sleep(1)


def _wait_fill_or_cancel(client, order_id, wait_fill_seconds: int) -> dict:
    """Waits for a market order to fill; on timeout CANCELS it instead of
    abandoning it. An abandoned DAY order still fills later (e.g. at the next
    open) with no stop and no local state -- the source of every orphaned,
    unprotected position on the v1/v2 accounts. Any quantity that filled
    before the cancel landed is still reported, so the caller protects it."""
    deadline = time.time() + wait_fill_seconds
    while time.time() < deadline:
        fetched = client.get_order_by_id(order_id)
        if fetched.status == "filled":
            return {"filled": True, "fill_price": float(fetched.filled_avg_price), "qty": float(fetched.filled_qty)}
        if fetched.status.value in TERMINAL_STATUSES:
            break
        time.sleep(2)
    fetched = _cancel_and_collect(client, order_id)
    filled_qty = float(fetched.filled_qty or 0)
    if filled_qty > 0:
        return {"filled": True, "fill_price": float(fetched.filled_avg_price), "qty": filled_qty}
    return {"filled": False}


def submit_market_buy(symbol: str, qty: float, wait_fill_seconds: int = 30) -> dict:
    """Returns {"filled": False, "rejected": True, "error": "..."} if Alpaca
    rejects the order outright (insufficient buying power, wash-trade
    protection, etc.) instead of letting the APIError propagate -- a single
    symbol's order getting rejected should never abort the rest of a live
    trading run (or skip save_state for every strategy that already
    succeeded this cycle), it should just be skipped like an unfilled order."""
    from alpaca.common.exceptions import APIError
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = trading_client()
    if not client.get_clock().is_open:
        # A market order outside RTH just queues for the next open -- long after
        # the run that sized it has given up waiting.
        return {"filled": False, "rejected": True, "error": "market closed"}
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=round(qty, 4), side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        ))
    except APIError as e:
        return {"filled": False, "rejected": True, "error": str(e)}

    return _wait_fill_or_cancel(client, order.id, wait_fill_seconds)


def submit_market_sell(symbol: str, qty: float, wait_fill_seconds: int = 30) -> dict:
    """Closes a long position at market -- used by PCSE's take-profit, which
    is checked periodically (has price touched the level since last check?)
    rather than resting as a second order alongside the stop-loss. Avoids
    ever having two simultaneous exit orders committing the same shares."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    from alpaca.common.exceptions import APIError

    client = trading_client()
    if not client.get_clock().is_open:
        return {"filled": False, "rejected": True, "error": "market closed"}
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=round(qty, 4), side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
        ))
    except APIError as e:
        return {"filled": False, "rejected": True, "error": str(e)}

    return _wait_fill_or_cancel(client, order.id, wait_fill_seconds)


def get_order_status(order_id: str) -> dict:
    """Used to reconcile per-strategy lots: Alpaca nets positions per symbol,
    so when two strategies hold the same symbol, checking whether a SPECIFIC
    stop order has filled is the only reliable way to know which strategy's
    lot closed."""
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


def cancel_order(order_id: str) -> dict:
    """Cancels and waits for confirmation (see _cancel_and_collect). Returns
    {"status": ...}, plus fill_price/fill_qty when the order FILLED before the
    cancel landed -- for a stop, that means the lot is already closed and the
    caller must not sell it again or re-place a stop."""
    fetched = _cancel_and_collect(trading_client(), order_id)
    out = {"status": fetched.status.value}
    if fetched.status == "filled":
        out["fill_price"] = float(fetched.filled_avg_price)
        out["fill_qty"] = float(fetched.filled_qty)
    return out
