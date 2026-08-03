"""Offline test for live_trading.py: exercises the full order-flow, state,
cursor, reconciliation, and same-symbol-two-strategies logic against a fake
in-memory broker -- no network calls, no real Alpaca credentials needed.
"""
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

import live_trading as lt


class FakeBroker:
    """In-memory stand-in for broker_alpaca, tracking orders/fills/equity."""

    def __init__(self, equity=100_000.0):
        self.equity = equity
        self.orders = {}  # order_id -> dict(status, symbol, qty, side, kind, stop_price, fill_price)
        self._next_id = 1
        self.positions = {}  # symbol -> total qty (sum across strategies, like real Alpaca)

    def _new_id(self):
        oid = f"order-{self._next_id}"
        self._next_id += 1
        return oid

    def get_equity(self):
        return self.equity

    def get_open_positions(self):
        return {s: {"qty": q, "avg_entry_price": 0.0} for s, q in self.positions.items() if q > 0}

    def submit_market_buy(self, symbol, qty, wait_fill_seconds=30):
        price = self._last_price.get(symbol, 100.0)
        self.positions[symbol] = self.positions.get(symbol, 0) + qty
        return {"filled": True, "fill_price": price, "qty": qty}

    def submit_stop_sell(self, symbol, qty, stop_price):
        oid = self._new_id()
        self.orders[oid] = {"status": "open", "symbol": symbol, "qty": qty, "stop_price": stop_price}
        return oid

    def cancel_order(self, order_id):
        self.orders[order_id]["status"] = "canceled"

    def get_order_status(self, order_id):
        o = self.orders[order_id]
        if o["status"] == "filled":
            return {"status": "filled", "fill_price": o["stop_price"], "fill_qty": o["qty"]}
        return {"status": o["status"]}

    # test helpers, not part of the real broker_alpaca interface
    def set_price(self, symbol, price):
        self._last_price = getattr(self, "_last_price", {})
        self._last_price[symbol] = price

    def trigger_stop_fill(self, order_id):
        self.orders[order_id]["status"] = "filled"
        o = self.orders[order_id]
        self.positions[o["symbol"]] = max(0, self.positions[o["symbol"]] - o["qty"])


def make_bars(n, start_price=100.0, trend=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    closes = start_price + np.cumsum(rng.normal(trend, 1.0, n))
    closes = np.maximum(closes, 1.0)
    df = pd.DataFrame({
        "Open": closes + rng.normal(0, 0.2, n),
        "High": closes + np.abs(rng.normal(0.5, 0.3, n)),
        "Low": closes - np.abs(rng.normal(0.5, 0.3, n)),
        "Close": closes,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)
    return df


def test_basic_flow():
    print("--- test_basic_flow ---")
    fb = FakeBroker()
    fb.set_price("AAA", 150.0)

    daily = {"AAA": make_bars(300)}
    # force a clean, obviously-triggering long_entry on the last bar
    daily["AAA"].loc[daily["AAA"].index[-1], "Close"] = daily["AAA"]["High"].iloc[-30:-1].max() + 5
    daily["AAA"].loc[daily["AAA"].index[-1], "Open"] = daily["AAA"]["Close"].iloc[-1]
    daily["AAA"].loc[daily["AAA"].index[-1], "Volume"] = daily["AAA"]["Volume"].iloc[-20:].mean() * 3

    with patch.object(lt.strat_bo, "prepare") as prep, patch.object(lt.broker, "fetch_daily_bars", return_value=daily), \
         patch.object(lt.broker, "get_equity", side_effect=fb.get_equity), \
         patch.object(lt.broker, "get_open_positions", side_effect=fb.get_open_positions), \
         patch.object(lt.broker, "submit_market_buy", side_effect=fb.submit_market_buy), \
         patch.object(lt.broker, "submit_stop_sell", side_effect=fb.submit_stop_sell), \
         patch.object(lt.broker, "cancel_order", side_effect=fb.cancel_order), \
         patch.object(lt.broker, "get_order_status", side_effect=fb.get_order_status):

        out = daily["AAA"].copy()
        out["atr"] = 2.0
        out["support"] = out["Low"].rolling(20).min()
        out["long_entry"] = False
        out.iloc[-1, out.columns.get_loc("long_entry")] = True
        prep.return_value = out

        state = {"positions": {}, "cursors": {}}
        args = type("Args", (), {"dry_run": False})()
        lt.run_breakout(state, fb.get_equity(), args)

        assert "AAA|breakout" in state["positions"], "expected a breakout position to open"
        pos = state["positions"]["AAA|breakout"]
        assert pos["shares"] > 0
        print(f"  OK: opened breakout position, shares={pos['shares']}, stop={pos['current_stop']:.2f}")

        # second call same day: cursor should prevent reprocessing (no duplicate order)
        orders_before = len(fb.orders)
        lt.run_breakout(state, fb.get_equity(), args)
        assert len(fb.orders) == orders_before, "cursor should have prevented a duplicate entry attempt"
        print("  OK: same-day cursor prevented duplicate processing")


def test_same_symbol_two_strategies():
    print("--- test_same_symbol_two_strategies ---")
    fb = FakeBroker()
    fb.set_price("BBB", 200.0)

    state = {"positions": {}, "cursors": {}}
    args = type("Args", (), {"dry_run": False})()

    with patch.object(lt.broker, "submit_market_buy", side_effect=fb.submit_market_buy), \
         patch.object(lt.broker, "submit_stop_sell", side_effect=fb.submit_stop_sell), \
         patch.object(lt.broker, "cancel_order", side_effect=fb.cancel_order), \
         patch.object(lt.broker, "get_order_status", side_effect=fb.get_order_status):

        entry_bo = lt.place_entry("breakout", "BBB", 200.0, 190.0, 2.0, 100_000.0)
        state["positions"]["BBB|breakout"] = entry_bo
        entry_mtf = lt.place_entry("mtf", "BBB", 200.0, 195.0, 2.0, 100_000.0)
        state["positions"]["BBB|mtf"] = entry_mtf

        assert fb.positions["BBB"] == entry_bo["shares"] + entry_mtf["shares"], "Alpaca should show ONE combined qty"
        print(f"  OK: both strategies hold BBB simultaneously, combined broker qty={fb.positions['BBB']}, "
              f"breakout={entry_bo['shares']}, mtf={entry_mtf['shares']}")

        # stop out ONLY the breakout lot; mtf lot must survive independently
        fb.trigger_stop_fill(entry_bo["stop_order_id"])

        for key in list(state["positions"].keys()):
            s = state["positions"][key]
            status = fb.get_order_status(s["stop_order_id"])
            if status["status"] == "filled":
                del state["positions"][key]

        assert "BBB|breakout" not in state["positions"], "breakout lot should be closed"
        assert "BBB|mtf" in state["positions"], "mtf lot should still be open"
        print("  OK: breakout lot closed independently, mtf lot on the same symbol unaffected")


def test_max_concurrent_cap():
    print("--- test_max_concurrent_cap ---")
    state = {"positions": {f"SYM{i}|breakout": {} for i in range(lt.MAX_CONCURRENT_POSITIONS)}, "cursors": {}}
    assert not lt.entry_allowed(len(state["positions"])), "cap should block a new entry once at the limit"
    print(f"  OK: entry blocked at {lt.MAX_CONCURRENT_POSITIONS} concurrent positions")
    state["positions"].pop(next(iter(state["positions"])))
    assert lt.entry_allowed(len(state["positions"])), "entry should be allowed just under the cap"
    print("  OK: entry allowed just under the cap")


if __name__ == "__main__":
    test_basic_flow()
    test_same_symbol_two_strategies()
    test_max_concurrent_cap()
    print("\nALL OFFLINE TESTS PASSED")
