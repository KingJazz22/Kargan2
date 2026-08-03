"""Breakout Hunter -- daily paper-trading runner.

Run ONCE per day, shortly after market open (e.g. 9:35am ET). Uses daily bars
through yesterday's close (final by then) to compute signals -- matches the
backtest's lookahead-safe convention (signal on close, execute at next open)
exactly, since "next open" from yesterday's signal IS today's open.

Scope/limitations vs. the backtest (read before trusting this with money):
  - The backtest ran each of the 40 symbols as an independent $100k account.
    This runner trades all 40 from ONE shared paper account/equity pool, which
    the backtest never modeled. If many symbols squeeze and break out
    together (plausible for correlated large-caps in a market-wide low-vol
    regime -- exactly when this strategy fires), position sizing computed
    independently per symbol could commit much more aggregate risk than any
    single-symbol backtest implied. MAX_CONCURRENT_POSITIONS below is a new
    safeguard not present in the backtest, added for this reason.
  - No commissions/slippage were modeled in the backtest; Alpaca paper fills
    are more realistic but still not identical to live execution costs.

Usage:
    python live_breakout.py            # live paper run, places real (paper) orders
    python live_breakout.py --dry-run  # compute and print signals/sizing only
"""
import argparse
from datetime import datetime

import broker_alpaca as broker
import strategy_breakout as strat
from basket import US_SYMBOLS
from state_store import load_state, save_state
from test_mtf_oos_symbols import OOS_US_SYMBOLS

SYMBOLS = US_SYMBOLS + OOS_US_SYMBOLS  # the 40 symbols the edge was validated on -- US only, no crypto
RISK_PCT = 0.01
MAX_CONCURRENT_POSITIONS = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compute signals/sizing only, place no orders")
    args = parser.parse_args()

    print(f"=== Breakout Hunter live run: {datetime.now().isoformat()} {'(DRY RUN)' if args.dry_run else ''} ===")

    state = load_state()
    equity = broker.get_equity()
    broker_positions = broker.get_open_positions()
    print(f"Account equity: ${equity:,.2f} | open positions at broker: {len(broker_positions)}")

    # reconcile: state entries no longer held at the broker were stopped/closed since last run
    for symbol in list(state.keys()):
        if symbol not in broker_positions:
            print(f"  {symbol}: no longer held at broker (stopped out or closed) -- clearing local state")
            del state[symbol]

    print("Fetching daily bars...")
    bars = broker.fetch_daily_bars(SYMBOLS)

    prepared = {}
    for symbol, df in bars.items():
        if len(df) < strat.PERCENTILE_LOOKBACK + 50:
            print(f"  skip {symbol}: not enough history ({len(df)} bars)")
            continue
        prepared[symbol] = strat.prepare(df)

    # 1. manage existing positions: update trailing stop
    for symbol, pos in broker_positions.items():
        if symbol not in prepared or symbol not in state:
            continue
        row = prepared[symbol].iloc[-1]
        s = state[symbol]

        s["extreme"] = max(s["extreme"], float(row["High"]))
        favorable = float(row["Close"]) - s["entry_price"]
        if not s["trail_active"] and favorable >= strat.TRAIL_ACTIVATE_ATR * s["atr_at_entry"]:
            s["trail_active"] = True
            print(f"  {symbol}: trailing stop ACTIVATED (favorable move {favorable:.2f} >= {strat.TRAIL_ACTIVATE_ATR}x ATR)")

        if s["trail_active"]:
            candidate = s["extreme"] - strat.TRAIL_ATR_MULT * float(row["atr"])
            current_stop = s.get("current_stop", s["initial_stop"])
            new_stop = max(current_stop, candidate)
            if new_stop > current_stop:
                print(f"  {symbol}: raising trail stop {current_stop:.2f} -> {new_stop:.2f}")
                if not args.dry_run:
                    broker.cancel_order(s["stop_order_id"])
                    s["stop_order_id"] = broker.submit_stop_sell(symbol, pos["qty"], new_stop)
                s["current_stop"] = new_stop

    # 2. new entries
    open_count = len(broker_positions)
    for symbol, data in prepared.items():
        if symbol in broker_positions:
            continue
        if open_count >= MAX_CONCURRENT_POSITIONS:
            continue
        row = data.iloc[-1]
        if not bool(row["long_entry"]):
            continue

        support = float(row["support"])
        atr_at_signal = float(row["atr"])
        approx_entry = float(row["Close"])  # yesterday's close, for a sizing estimate -- actual fill may differ slightly
        initial_stop = support - strat.INITIAL_STOP_ATR_BUFFER * atr_at_signal
        stop_distance = approx_entry - initial_stop
        if stop_distance <= 0:
            continue

        risk_amount = equity * RISK_PCT
        qty = risk_amount / stop_distance
        qty = min(qty, equity / approx_entry)
        # whole shares only: Alpaca stop orders don't support fractional quantities
        qty = int(qty)
        if qty < 1:
            continue

        print(f"  {symbol}: ENTRY signal. approx_entry={approx_entry:.2f} initial_stop={initial_stop:.2f} qty={qty:.4f}")
        if args.dry_run:
            open_count += 1
            continue

        fill = broker.submit_market_buy(symbol, qty)
        if not fill["filled"]:
            print(f"    {symbol}: order not filled within wait window, skipping stop placement")
            continue

        entry_price = fill["fill_price"]
        filled_qty = fill["qty"]
        emergency_stop = initial_stop - strat.EMERGENCY_STOP_BUFFER_ATR * atr_at_signal
        stop_order_id = broker.submit_stop_sell(symbol, filled_qty, initial_stop)

        state[symbol] = {
            "entry_date": str(data.index[-1]),
            "entry_price": entry_price,
            "atr_at_entry": atr_at_signal,
            "initial_stop": initial_stop,
            "emergency_stop": emergency_stop,
            "current_stop": initial_stop,
            "trail_active": False,
            "extreme": entry_price,
            "stop_order_id": stop_order_id,
        }
        open_count += 1
        print(f"    filled @ {entry_price:.2f} qty={filled_qty} -- stop placed @ {initial_stop:.2f}")

    if not args.dry_run:
        save_state(state)
    print("=== done ===")


if __name__ == "__main__":
    main()
