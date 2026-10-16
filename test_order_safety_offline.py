"""Offline regression tests for the order-safety fixes that left live
positions unprotected (Oct 2026):

  * an entry order that timed out was abandoned, not cancelled -> it filled at
    the next open with no stop and no lot (META/PG/INTC/QCOM/AMZN/QQQ on v1,
    XOM/WMT on v2);
  * a take-profit sold right after a bare (unconfirmed) stop cancel -> refused,
    run aborted, next run dropped the lot -> HD held with no stop (v2);
  * reconciliation dropped any lot whose stop was cancelled, even while the
    shares were still held.

Runs both live_trading (v1) and live_trading_v2 against the in-memory
FakeBroker from test_live_trading_offline, plus the broker helpers against a
fake Alpaca client. No network, no credentials used.
"""
from types import SimpleNamespace
from unittest.mock import patch

import broker_alpaca
import broker_alpaca_v2
import live_trading as lt1
import live_trading_v2 as lt2
from test_live_trading_offline import FakeBroker, make_bars

ARGS = SimpleNamespace(dry_run=False)


class RefusingFakeBroker(FakeBroker):
    """Market sells are refused (e.g. market closed / shares still reserved)."""

    def submit_market_sell(self, symbol, qty, wait_fill_seconds=30):
        return {"filled": False, "rejected": True, "error": "market closed"}


def _pcse_lot(fb, symbol="CCC", shares=5, stop=90.0):
    fb.orders["stop-1"] = {"status": "open", "symbol": symbol, "qty": shares, "stop_price": stop}
    fb.positions[symbol] = shares
    return {"entry_price": 100.0, "shares": shares, "atr_at_entry": 1.0, "current_stop": stop,
            "initial_stop": stop, "trail_active": False, "extreme": 100.0, "stop_order_id": "stop-1"}


def _run_pcse_tp(lt, fb, state):
    """Drives run_pcse with a bar whose High touches the BB take-profit (106)."""
    bars = {"CCC": make_bars(lt.PCSE_BARS_NEEDED + 10, start_price=100.0)}
    out = bars["CCC"].copy()
    out["bb_mid"], out["bb_std"], out["long_entry"] = 100.0, 2.0, False
    out["ret_stdev"] = 0.01
    out.iloc[-1, out.columns.get_loc("High")] = 110.0
    patches = [patch.object(lt, "PCSE_SYMBOLS", ["CCC"]),
               patch.object(lt.strat_pcse, "prepare", return_value=out),
               patch.object(lt.broker, "cancel_order", side_effect=fb.cancel_order),
               patch.object(lt.broker, "submit_market_sell", side_effect=fb.submit_market_sell),
               patch.object(lt.broker, "submit_stop_sell", side_effect=fb.submit_stop_sell)]
    if lt is lt1:
        patches.append(patch.object(lt.broker, "fetch_4h_bars", return_value=bars))
    else:
        patches.append(patch.object(lt.bar_cache_v2, "fetch_4h_bars_cached", return_value=bars))
    for p in patches:
        p.start()
    try:
        lt.run_pcse(state, fb.get_equity(), lt.RISK_PCT, False, ARGS)
    finally:
        for p in patches:
            p.stop()


def test_tp_when_stop_already_filled(lt):
    fb = FakeBroker()
    fb.set_price("CCC", 107.0)
    state = {"positions": {"CCC|pcse": _pcse_lot(fb)}, "cursors": {}, "trade_log": []}
    fb.trigger_stop_fill("stop-1")  # stop fills between the High check and our cancel
    _run_pcse_tp(lt, fb, state)
    assert "CCC|pcse" not in state["positions"]
    assert fb.positions["CCC"] == 0, "must not sell shares the stop already sold"
    assert len(state["trade_log"]) == 1 and state["trade_log"][0]["win"] is False, "logged at the stop price"
    print(f"  OK [{lt.__name__}]: TP after a stop fill -> no double sell, logged at stop price")


def test_tp_sell_refused_keeps_lot_protected(lt):
    fb = RefusingFakeBroker()
    fb.set_price("CCC", 107.0)
    state = {"positions": {"CCC|pcse": _pcse_lot(fb)}, "cursors": {}, "trade_log": []}
    _run_pcse_tp(lt, fb, state)
    s = state["positions"].get("CCC|pcse")
    assert s is not None, "lot must be kept when the exit sell didn't fill"
    assert s["stop_order_id"] != "stop-1" and fb.orders[s["stop_order_id"]]["status"] == "open", \
        "a fresh stop must protect the shares"
    assert fb.orders[s["stop_order_id"]]["qty"] == 5 and not state["trade_log"]
    print(f"  OK [{lt.__name__}]: refused TP sell -> lot kept, stop re-placed")


def test_reconcile_replaces_cancelled_stop(lt):
    fb = FakeBroker()
    state = {"positions": {"HD|pcse": _pcse_lot(fb, "HD", shares=8, stop=280.0)}, "trade_log": [],
             "orland_trade_log": []}
    fb.orders["stop-1"]["status"] = "canceled"
    with patch.object(lt.broker, "get_order_status", side_effect=fb.get_order_status), \
         patch.object(lt.broker, "submit_stop_sell", side_effect=fb.submit_stop_sell):
        lt.reconcile(state, fb.get_open_positions(), ARGS)
    s = state["positions"].get("HD|pcse")
    assert s is not None and fb.orders[s["stop_order_id"]]["status"] == "open"
    assert fb.orders[s["stop_order_id"]]["stop_price"] == 280.0
    print(f"  OK [{lt.__name__}]: cancelled stop with shares still held -> re-placed, lot kept")


def test_reconcile_drops_lot_when_shares_gone(lt):
    fb = FakeBroker()
    state = {"positions": {"HD|pcse": _pcse_lot(fb, "HD", shares=8)}, "trade_log": []}
    fb.orders["stop-1"]["status"] = "canceled"
    fb.positions["HD"] = 0
    with patch.object(lt.broker, "get_order_status", side_effect=fb.get_order_status), \
         patch.object(lt.broker, "submit_stop_sell", side_effect=AssertionError("must not re-place")):
        lt.reconcile(state, fb.get_open_positions(), ARGS)
    assert "HD|pcse" not in state["positions"]
    print(f"  OK [{lt.__name__}]: cancelled stop with no shares held -> lot cleared")


def test_reconcile_flags_untracked_shares(lt, capsys=None):
    import io, contextlib
    fb = FakeBroker()
    state = {"positions": {}, "trade_log": []}
    fb.positions["AMZN"] = 24
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lt.reconcile(state, fb.get_open_positions(), ARGS)
    assert "AMZN: 24 shares held but not tracked" in buf.getvalue()
    print(f"  OK [{lt.__name__}]: untracked broker shares are flagged")


class FakeOrder:
    def __init__(self, status, filled_qty=0, price=None):
        self.status = SimpleNamespace(value=status)
        self.filled_qty, self.filled_avg_price = filled_qty, price


class FakeOrder(FakeOrder):
    def __init__(self, status, filled_qty=0, price=None):
        super().__init__(status, filled_qty, price)

        class S(str):
            value = status
        self.status = S(status)


class FakeClient:
    """Alpaca TradingClient stand-in: an order that never fills on its own,
    with a configurable outcome once cancelled."""

    def __init__(self, open_=True, after_cancel=("canceled", 0, None)):
        self.open_, self.after_cancel = open_, after_cancel
        self.cancelled, self.submitted = [], []

    def get_clock(self):
        return SimpleNamespace(is_open=self.open_)

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id="o1")

    def get_order_by_id(self, oid):
        if oid in self.cancelled:
            return FakeOrder(*self.after_cancel)
        return FakeOrder("new")

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)


def test_broker_timeout_cancels(broker):
    fc = FakeClient()
    with patch.object(broker, "trading_client", return_value=fc), patch.object(broker.time, "sleep"):
        res = broker.submit_market_buy("XOM", 3, wait_fill_seconds=0)
    assert fc.cancelled == ["o1"] and res == {"filled": False}
    print(f"  OK [{broker.__name__}]: timed-out entry is cancelled, not abandoned")


def test_broker_timeout_reports_late_fill(broker):
    fc = FakeClient(after_cancel=("filled", 3, 176.71))
    with patch.object(broker, "trading_client", return_value=fc), patch.object(broker.time, "sleep"):
        res = broker.submit_market_buy("XOM", 3, wait_fill_seconds=0)
    assert res == {"filled": True, "fill_price": 176.71, "qty": 3.0}, res
    print(f"  OK [{broker.__name__}]: fill that races the cancel is reported (so it gets a stop)")


def test_broker_market_closed(broker):
    fc = FakeClient(open_=False)
    with patch.object(broker, "trading_client", return_value=fc):
        buy = broker.submit_market_buy("AMZN", 12)
        sell = broker.submit_market_sell("AMZN", 12)
    assert not fc.submitted and buy["rejected"] and sell["rejected"]
    print(f"  OK [{broker.__name__}]: no market orders submitted while the market is closed")


def test_broker_cancel_reports_stop_fill(broker):
    fc = FakeClient(after_cancel=("filled", 8, 297.5))
    with patch.object(broker, "trading_client", return_value=fc), patch.object(broker.time, "sleep"):
        res = broker.cancel_order("stop-hd")
    assert res == {"status": "filled", "fill_price": 297.5, "fill_qty": 8.0}, res
    print(f"  OK [{broker.__name__}]: cancel_order reports a stop that filled first")


if __name__ == "__main__":
    for lt in (lt1, lt2):
        print(f"--- {lt.__name__} ---")
        test_tp_when_stop_already_filled(lt)
        test_tp_sell_refused_keeps_lot_protected(lt)
        test_reconcile_replaces_cancelled_stop(lt)
        test_reconcile_drops_lot_when_shares_gone(lt)
        test_reconcile_flags_untracked_shares(lt)
    for b in (broker_alpaca, broker_alpaca_v2):
        print(f"--- {b.__name__} ---")
        test_broker_timeout_cancels(b)
        test_broker_timeout_reports_late_fill(b)
        test_broker_market_closed(b)
        test_broker_cancel_reports_stop_fill(b)
    print("\nALL ORDER-SAFETY TESTS PASSED")
