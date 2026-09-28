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

    def submit_market_sell(self, symbol, qty, wait_fill_seconds=30):
        price = self._last_price.get(symbol, 100.0)
        self.positions[symbol] = max(0, self.positions.get(symbol, 0) - qty)
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
        lt.run_breakout(state, fb.get_equity(), lt.RISK_PCT, False, args)

        assert "AAA|breakout" in state["positions"], "expected a breakout position to open"
        pos = state["positions"]["AAA|breakout"]
        assert pos["shares"] > 0
        print(f"  OK: opened breakout position, shares={pos['shares']}, stop={pos['current_stop']:.2f}")

        # second call same day: cursor should prevent reprocessing (no duplicate order)
        orders_before = len(fb.orders)
        lt.run_breakout(state, fb.get_equity(), lt.RISK_PCT, False, args)
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

        entry_bo = lt.place_entry("breakout", "BBB", 200.0, 190.0, 2.0, 100_000.0, lt.RISK_PCT)
        state["positions"]["BBB|breakout"] = entry_bo
        entry_mtf = lt.place_entry("mtf", "BBB", 200.0, 195.0, 2.0, 100_000.0, lt.RISK_PCT)
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


def test_pcse_take_profit_close():
    print("--- test_pcse_take_profit_close ---")
    fb = FakeBroker()
    fb.set_price("CCC", 150.0)

    daily = {"CCC": make_bars(lt.PCSE_BARS_NEEDED + 10, start_price=100.0)}

    state = {"positions": {}, "cursors": {}}
    args = type("Args", (), {"dry_run": False})()

    entry_price = 100.0
    stop_order_id = "stop-1"
    fb.orders[stop_order_id] = {"status": "open", "symbol": "CCC", "qty": 5, "stop_price": 90.0}
    fb.positions["CCC"] = 5
    state["positions"]["CCC|pcse"] = {
        "entry_price": entry_price, "shares": 5, "atr_at_entry": 1.0,
        "current_stop": 90.0, "initial_stop": 90.0, "trail_active": False, "extreme": entry_price,
        "stop_order_id": stop_order_id,
    }

    with patch.object(lt, "PCSE_SYMBOLS", ["CCC"]), \
         patch.object(lt.strat_pcse, "prepare") as prep, \
         patch.object(lt.broker, "fetch_4h_bars", return_value=daily), \
         patch.object(lt.broker, "cancel_order", side_effect=fb.cancel_order), \
         patch.object(lt.broker, "submit_market_sell", side_effect=fb.submit_market_sell), \
         patch.object(lt.broker, "get_order_status", side_effect=fb.get_order_status):

        out = daily["CCC"].copy()
        out["bb_mid"] = 100.0
        out["bb_std"] = 2.0  # tp_price = 100 + BB_NUM_STD(3.0)*2.0 = 106
        out["long_entry"] = False
        out.iloc[-1, out.columns.get_loc("High")] = 110.0  # touches tp_price (106)
        prep.return_value = out

        lt.run_pcse(state, fb.get_equity(), lt.RISK_PCT, False, args)

        assert "CCC|pcse" not in state["positions"], "PCSE position should have closed on take-profit touch"
        assert fb.orders[stop_order_id]["status"] == "canceled", "stop order should have been canceled"
        assert fb.positions["CCC"] == 0, "shares should have been sold"
        print("  OK: PCSE take-profit touch closed the position and canceled the stop")
        assert len(state["trade_log"]) == 1, "take-profit close should have logged one trade"
        assert state["trade_log"][0]["win"] is True, "closing above entry should log a win"


def test_halt_entries_blocks_new_positions():
    print("--- test_halt_entries_blocks_new_positions ---")
    fb = FakeBroker()
    fb.set_price("AAA", 150.0)

    daily = {"AAA": make_bars(300)}
    daily["AAA"].loc[daily["AAA"].index[-1], "Close"] = daily["AAA"]["High"].iloc[-30:-1].max() + 5
    daily["AAA"].loc[daily["AAA"].index[-1], "Open"] = daily["AAA"]["Close"].iloc[-1]
    daily["AAA"].loc[daily["AAA"].index[-1], "Volume"] = daily["AAA"]["Volume"].iloc[-20:].mean() * 3

    with patch.object(lt.strat_bo, "prepare") as prep, patch.object(lt.broker, "fetch_daily_bars", return_value=daily), \
         patch.object(lt.broker, "submit_market_buy", side_effect=fb.submit_market_buy):

        out = daily["AAA"].copy()
        out["atr"] = 2.0
        out["support"] = out["Low"].rolling(20).min()
        out["long_entry"] = False
        out.iloc[-1, out.columns.get_loc("long_entry")] = True
        prep.return_value = out

        state = {"positions": {}, "cursors": {}}
        args = type("Args", (), {"dry_run": False})()
        lt.run_breakout(state, fb.get_equity(), lt.RISK_PCT, True, args)  # halt_entries=True

        assert "AAA|breakout" not in state["positions"], "a valid signal must NOT open a position while entries are halted"
        assert len(fb.orders) == 0, "no order should have been placed while halted"
        print("  OK: halt_entries=True suppressed a real entry signal")


def test_risk_throttle():
    print("--- test_risk_throttle ---")
    state = {"trade_log": []}
    risk = lt.update_risk_throttle(state, equity=10_000.0, peak_equity=10_000.0)
    assert risk == lt.RISK_PCT, "no trade history yet -- should size at full risk"

    # 3 consecutive losses should throttle risk down
    state["trade_log"] = [{"r": -1.0, "win": False}] * 3
    risk = lt.update_risk_throttle(state, equity=9_700.0, peak_equity=10_000.0)
    assert risk == pytest_approx(lt.RISK_PCT * lt.RISK_THROTTLE_MULT), "3 losses in a row should throttle risk"

    # a new equity high should clear the throttle even with the same losing history
    risk = lt.update_risk_throttle(state, equity=10_050.0, peak_equity=10_050.0)
    assert risk == lt.RISK_PCT, "a new equity peak should restore full risk"
    print("  OK: risk throttle engages on 3 losses, clears on a new equity peak")


def pytest_approx(x, tol=1e-9):
    """Tiny local float-compare helper -- avoids pulling in pytest for one assertion."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_partial_profit_take():
    print("--- test_partial_profit_take ---")
    fb = FakeBroker()
    fb.set_price("DDD", 110.0)

    stop_order_id = "stop-partial"
    fb.orders[stop_order_id] = {"status": "open", "symbol": "DDD", "qty": 10, "stop_price": 95.0}
    fb.positions["DDD"] = 10
    s = {
        "entry_price": 100.0, "shares": 10, "atr_at_entry": 1.0, "current_stop": 95.0,
        "initial_stop": 95.0, "trail_active": True, "extreme": 110.0,
        "stop_order_id": stop_order_id, "partial_taken": False,
    }

    with patch.object(lt.broker, "cancel_order", side_effect=fb.cancel_order), \
         patch.object(lt.broker, "submit_market_sell", side_effect=fb.submit_market_sell), \
         patch.object(lt.broker, "submit_stop_sell", side_effect=fb.submit_stop_sell):

        args = type("Args", (), {"dry_run": False})()
        # entry=100, stop=95 -> 1R=5; price=110 -> +2.0R, above PARTIAL_PROFIT_R (1.2)
        lt.check_partial_profit("DDD", "breakout", s, 110.0, args)

        assert s["partial_taken"] is True
        assert s["shares"] == 7, f"expected 30% of 10 shares (3) sold, 7 remaining, got {s['shares']}"
        assert fb.orders[stop_order_id]["status"] == "canceled", "original stop should be canceled"
        assert fb.positions["DDD"] == 7, "broker position should reflect the partial sell"
        new_stop_orders = [o for o in fb.orders.values() if o["symbol"] == "DDD" and o["status"] == "open"]
        assert len(new_stop_orders) == 1 and new_stop_orders[0]["qty"] == 7, "a fresh stop for the reduced qty should be resting"
        print(f"  OK: partial profit sold 3/10 shares at +2.0R, re-rested stop for {s['shares']}")

        # calling again on the same lot should be a no-op (one-shot per lot)
        orders_before = len(fb.orders)
        lt.check_partial_profit("DDD", "breakout", s, 115.0, args)
        assert len(fb.orders) == orders_before, "partial profit should only trigger once per lot"
        print("  OK: partial profit is one-shot per lot")


def _make_orland_timeframe_data():
    return {
        "1D": make_bars(300, start_price=30.0, seed=11),
        "1W": make_bars(120, start_price=30.0, seed=12),
        "30D": make_bars(60, start_price=30.0, seed=13),
        "1H": make_bars(300, start_price=30.0, seed=14),
        "15m": make_bars(300, start_price=30.0, seed=15),
        "5m": make_bars(300, start_price=30.0, seed=16),
    }


def test_orland_s1_entry_and_risk_scoping():
    print("--- test_orland_s1_entry_and_risk_scoping ---")
    fb = FakeBroker()
    fb.set_price("EWZ", 30.0)

    timeframe_data = _make_orland_timeframe_data()
    forced_signal = lt.strategy_s1.Signal(
        symbol="EWZ", direction="up", entry_price=30.0, stop=28.0, target=34.0,
        risk=2.0, mtf_score=2.0, intensity="STRONG", confirmations={},
    )

    state = {"positions": {}, "cursors": {}}
    args = type("Args", (), {"dry_run": False})()

    with patch.object(lt.orland_data, "get_multi_timeframe", return_value=timeframe_data), \
         patch.object(lt.strategy_s1, "check_entry", return_value=forced_signal), \
         patch.object(lt.broker, "submit_market_buy", side_effect=fb.submit_market_buy), \
         patch.object(lt.broker, "submit_stop_sell", side_effect=fb.submit_stop_sell):

        lt.run_orland_s1(state, equity=100_000.0, halt_entries=False, args=args)

        assert "EWZ|orland_s1" in state["positions"], "expected an orland_s1 position to open"
        pos = state["positions"]["EWZ|orland_s1"]
        # risk_amount = equity * ORLAND_RISK_PCT (half of RISK_PCT), stop_distance = 30-28 = 2
        expected_qty = int(min((100_000.0 * lt.ORLAND_RISK_PCT) / 2.0, 100_000.0 / 30.0))
        assert pos["shares"] == expected_qty, \
            f"expected qty sized off ORLAND_RISK_PCT ({lt.ORLAND_RISK_PCT}), got {pos['shares']} vs {expected_qty}"
        print(f"  OK: orland_s1 opened sized off ORLAND_RISK_PCT (half of RISK_PCT), shares={pos['shares']}")

        # halt_entries=True must suppress the same signal
        state2 = {"positions": {}, "cursors": {}}
        lt.run_orland_s1(state2, equity=100_000.0, halt_entries=True, args=args)
        assert "EWZ|orland_s1" not in state2["positions"], "orland halt must suppress a valid signal"
        print("  OK: orland_s1 respects halt_entries")


def test_orland_elder_entry():
    print("--- test_orland_elder_entry ---")
    fb = FakeBroker()
    fb.set_price("EWZ", 30.0)

    daily = {"EWZ": make_bars(800, start_price=30.0, seed=17)}
    forced_signal = lt.elder_strategy.ElderSignal(
        direction="up", entry_price=30.0, stop=28.0, target=34.0, risk=2.0,
    )

    state = {"positions": {}, "cursors": {}}
    args = type("Args", (), {"dry_run": False})()

    with patch.object(lt.broker, "fetch_daily_bars", return_value=daily), \
         patch.object(lt.elder_strategy, "check_entry", return_value=forced_signal), \
         patch.object(lt.broker, "submit_market_buy", side_effect=fb.submit_market_buy), \
         patch.object(lt.broker, "submit_stop_sell", side_effect=fb.submit_stop_sell):

        lt.run_orland_elder(state, equity=100_000.0, halt_entries=False, args=args)

        assert "EWZ|orland_elder" in state["positions"], "expected an orland_elder position to open"
        pos = state["positions"]["EWZ|orland_elder"]
        expected_qty = int(min((100_000.0 * lt.ORLAND_RISK_PCT) / 2.0, 100_000.0 / 30.0))
        assert pos["shares"] == expected_qty, \
            f"expected qty sized off ORLAND_RISK_PCT ({lt.ORLAND_RISK_PCT}), got {pos['shares']} vs {expected_qty}"
        print(f"  OK: orland_elder opened sized off ORLAND_RISK_PCT, shares={pos['shares']}")


def test_orland_consecutive_loss_halt():
    print("--- test_orland_consecutive_loss_halt ---")
    state = {"trade_log": [], "orland_trade_log": []}

    # 3 consecutive orland losses -> orland-specific halt fires
    state["orland_trade_log"] = [{"r": -1.0, "win": False}] * lt.ORLAND_CONSEC_LOSS_HALT
    assert lt.orland_entries_halted(state) is True, "3 consecutive orland losses should halt new orland entries"

    # main risk_throttle/trade_log stay completely untouched by orland's losses
    assert state["trade_log"] == [], "orland losses must never appear in the main trade_log"
    main_risk = lt.update_risk_throttle(state, equity=10_000.0, peak_equity=10_000.0)
    assert main_risk == lt.RISK_PCT, "orland losses must not throttle breakout/mtf/pcse's risk_pct"

    # a win clears the halt (self-clearing sliding window, no separate reset needed)
    state["orland_trade_log"].append({"r": 1.0, "win": True})
    assert lt.orland_entries_halted(state) is False, "a win should clear the orland halt"
    print("  OK: orland consecutive-loss halt is isolated from the main risk_throttle/trade_log, and self-clears on a win")


def test_reconciliation_routes_orland_stopouts_to_orland_log():
    print("--- test_reconciliation_routes_orland_stopouts_to_orland_log ---")
    fb = FakeBroker()
    fb.set_price("EWZ", 30.0)

    stop_order_id = "stop-orland"
    fb.orders[stop_order_id] = {"status": "filled", "symbol": "EWZ", "qty": 10, "stop_price": 28.0}
    state = {"positions": {}, "cursors": {}, "trade_log": [], "orland_trade_log": []}
    state["positions"]["EWZ|orland_s1"] = {
        "entry_price": 30.0, "shares": 10, "atr_at_entry": 1.0, "current_stop": 28.0,
        "initial_stop": 28.0, "trail_active": False, "extreme": 30.0,
        "stop_order_id": stop_order_id, "direction": "up",
    }

    with patch.object(lt.broker, "get_order_status", side_effect=fb.get_order_status):
        # mirrors main()'s reconciliation loop
        for key in list(state["positions"].keys()):
            s = state["positions"][key]
            status = fb.get_order_status(s["stop_order_id"])
            if status["status"] == "filled":
                log_key = "orland_trade_log" if key.endswith(("|orland_s1", "|orland_elder")) else "trade_log"
                lt.record_closed_trade(state, s, status["fill_price"], log_key=log_key)
                del state["positions"][key]

    assert len(state["orland_trade_log"]) == 1, "orland stop-out should log to orland_trade_log"
    assert len(state["trade_log"]) == 0, "orland stop-out must NOT touch the main trade_log"
    print("  OK: orland stop-outs route to orland_trade_log, not the main trade_log")


if __name__ == "__main__":
    test_basic_flow()
    test_same_symbol_two_strategies()
    test_pcse_take_profit_close()
    test_halt_entries_blocks_new_positions()
    test_risk_throttle()
    test_partial_profit_take()
    test_orland_s1_entry_and_risk_scoping()
    test_orland_elder_entry()
    test_orland_consecutive_loss_halt()
    test_reconciliation_routes_orland_stopouts_to_orland_log()
    print("\nALL OFFLINE TESTS PASSED")
