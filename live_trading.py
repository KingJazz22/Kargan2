"""Unified live/paper trading runner: Breakout Hunter (daily) + Multi-Timeframe
RSI+Stochastic v2 (4H execution + Daily confirm, TSL-only exit, VWAP(200)
trend filter) + PCSE (4H, SL=12x/TP=BB(3.0std)) sharing ONE Alpaca paper
account -- the three validated strategies from this project (Grid-Martingale
deliberately excluded -- its own FINDINGS_GRID_MARTINGALE.md says "still not
ready for live paper trading").

Run cadence:
  - Breakout Hunter only needs checking once per day (its signal is decided
    from the completed daily bar).
  - MTF and PCSE need checking roughly every 4 hours during market hours,
    since their execution timeframe is 4H bars -- running once/day at open
    would miss most of their signal timing.
  Schedule this script to run every ~2 hours during market hours (e.g.
  9:35, 11:35, 13:35, 15:35 ET) -- on Railway via railway_runner.py's internal
  scheduler. Breakout Hunter's daily logic uses a cursor to only act once per
  day even though the script runs more often; MTF/PCSE's logic uses a cursor
  to only act on a newly-completed 4H bar since the last run.

Real-account constraint the backtest didn't have: Alpaca nets positions per
SYMBOL at the account level -- it will only ever report ONE combined qty for
a symbol, even if both strategies hold it. This runner still lets both
strategies trade the same symbol simultaneously (matching backtest_combined.py),
by keeping each strategy's shares/entry/stop tracked entirely in LOCAL state
and placing a SEPARATE stop-sell order per (symbol, strategy) lot -- Alpaca
just needs the combined open sell quantity across both orders to not exceed
the held position, which it will always satisfy since each order only covers
its own strategy's shares. Reconciliation checks each position's specific
stop ORDER status (not "does the symbol still show a position"), since two
lots on the same symbol can close independently of each other.

Usage:
    python live_trading.py            # live paper run, places real (paper) orders
    python live_trading.py --dry-run  # compute and print signals/sizing only
"""
import argparse
from datetime import datetime

import pandas as pd

import broker_alpaca as broker
import strategy_breakout as strat_bo
import strategy_mtf as strat_mtf
import strategy_pcse_final as strat_pcse
from basket import US_SYMBOLS
from state_store import load_state, save_state
from test_mtf_oos_symbols import OOS_US_SYMBOLS

SYMBOLS = US_SYMBOLS + OOS_US_SYMBOLS  # the 40 symbols both edges were validated on
RISK_PCT = 0.01
# No position-count cap: simulate_full_kargan2_system.py's sweep showed cash
# availability (each entry sized off available cash, see place_entry) was
# already the real constraint -- capping at 10 vs 15/20/uncapped changed
# total return by well under 1pp and left max drawdown identical (-16.1%
# every time), consistent with this project's other combined-portfolio
# findings that position-COUNT caps stop binding long before they matter.

# PCSE was only validated on basket.US_SYMBOLS (20 symbols, the in-sample half
# of SYMBOLS above) -- NOT the OOS_US_SYMBOLS half, and not extended here
# beyond what FINDINGS_CANDLE_PROB.md actually tested.
PCSE_SYMBOLS = US_SYMBOLS
PCSE_BARS_NEEDED = strat_pcse.entry_strat.TRAIN_BARS + strat_pcse.entry_strat.TEST_BARS  # 1750
PCSE_LOOKBACK_DAYS = 800  # comfortably covers 1750+ 4H bars (~3/day for equities)

MTF_PARAMS = {
    "long_only": True, "htf_bar_hours": 24, "tsl_only_exit": True,
    # 200-day VWAP trend filter: only take longs above it. Added after a deep
    # (10.5yr, Alpaca) backtest showed unfiltered MTF v2 is a net loser
    # (t=-2.86 out-of-sample) -- it's a pure countertrend/oversold-bounce
    # strategy with no protection in real downtrends. A 200-day SMA filter
    # looked like a bigger fix in-sample but completely failed to replicate
    # out-of-sample (t collapsed from +1.23 to +0.14 -- overfit). VWAP(200)
    # was the only filter of 12 tested (SMA/EMA/VWAP x 50/100/150/200) that
    # held up on fresh out-of-sample symbols (t=+1.20 in-sample -> +0.62 OOS,
    # still same sign and modest positive avg R both times). See
    # sweep_mtf_trend_filter.py / FINDINGS_MTF.md for the full sweep.
    "trend_filter_period": 200, "trend_filter_bar_hours": 24, "trend_filter_type": "vwap",
}


def place_entry(strategy, symbol, qty_hint_price, initial_stop, atr_at_signal, equity, extra=None):
    """Submits the market buy, waits for fill, places the protective stop,
    and returns the new state entry (or None if not filled/sizeable)."""
    stop_distance = qty_hint_price - initial_stop
    if stop_distance <= 0:
        return None
    risk_amount = equity * RISK_PCT
    qty = int(min(risk_amount / stop_distance, equity / qty_hint_price))
    if qty < 1:
        return None

    fill = broker.submit_market_buy(symbol, qty)
    if not fill["filled"]:
        print(f"    {symbol}/{strategy}: order not filled within wait window, skipping")
        return None

    entry_price = fill["fill_price"]
    filled_qty = fill["qty"]
    stop_order_id = broker.submit_stop_sell(symbol, filled_qty, initial_stop)
    print(f"    {symbol}/{strategy}: filled @ {entry_price:.2f} qty={filled_qty} -- stop @ {initial_stop:.2f}")

    entry = {
        "entry_price": entry_price, "shares": filled_qty, "atr_at_entry": atr_at_signal,
        "current_stop": initial_stop, "trail_active": False, "extreme": entry_price,
        "stop_order_id": stop_order_id,
    }
    if extra:
        entry.update(extra)
    return entry


def update_trail(symbol, strategy, s, row, extreme_price, trail_mult, activate_atr, args):
    """activate_atr=None means the trail is unconditionally active from bar 1
    (MTF, per backtest_mtf.py -- no activation gate at all). A numeric
    activate_atr means the trail only takes over once favorable movement
    reaches that many ATRs of profit (breakout's two-stage stop).

    Uses s["shares"] (this lot's OWN tracked quantity), not the broker's
    combined position for the symbol -- another strategy may hold shares of
    the same symbol too."""
    pos_qty = s["shares"]
    s["extreme"] = max(s["extreme"], extreme_price)
    favorable = float(row["Close"]) - s["entry_price"]
    if activate_atr is None:
        s["trail_active"] = True
    elif not s["trail_active"] and favorable >= activate_atr * s["atr_at_entry"]:
        s["trail_active"] = True
        print(f"  {symbol}/{strategy}: trailing stop ACTIVATED")

    if s["trail_active"]:
        candidate = s["extreme"] - trail_mult * float(row["atr"])
        new_stop = max(s["current_stop"], candidate)
        if new_stop > s["current_stop"]:
            print(f"  {symbol}/{strategy}: raising stop {s['current_stop']:.2f} -> {new_stop:.2f}")
            if not args.dry_run:
                broker.cancel_order(s["stop_order_id"])
                s["stop_order_id"] = broker.submit_stop_sell(symbol, pos_qty, new_stop)
            s["current_stop"] = new_stop


def run_breakout(state, equity, args):
    print("\n--- Breakout Hunter (daily) ---")
    bars = broker.fetch_daily_bars(SYMBOLS)

    for symbol, df in bars.items():
        if len(df) < strat_bo.PERCENTILE_LOOKBACK + 50:
            continue
        prepared = strat_bo.prepare(df)
        row = prepared.iloc[-1]
        today_str = str(prepared.index[-1].date())
        cursor = state["cursors"].setdefault(symbol, {})
        key = f"{symbol}|breakout"

        # manage existing position (breakout tracks its extreme via High, matching backtest_breakout.py)
        if key in state["positions"]:
            s = state["positions"][key]
            update_trail(symbol, "breakout", s, row, float(row["High"]), strat_bo.TRAIL_ATR_MULT, strat_bo.TRAIL_ACTIVATE_ATR, args)

        if cursor.get("last_daily_date") == today_str:
            continue  # already processed today's bar
        cursor["last_daily_date"] = today_str

        if key not in state["positions"] and bool(row["long_entry"]):
            support, atr_sig = float(row["support"]), float(row["atr"])
            initial_stop = support - strat_bo.INITIAL_STOP_ATR_BUFFER * atr_sig
            print(f"  {symbol}: BREAKOUT entry signal")
            if args.dry_run:
                continue
            entry = place_entry("breakout", symbol, float(row["Close"]), initial_stop, atr_sig, equity)
            if entry:
                entry["initial_stop"] = initial_stop
                state["positions"][key] = entry


def run_mtf(state, equity, args):
    print("\n--- MTF v2 (4H+Daily, TSL-only exit, VWAP(200) trend filter) ---")
    ltf_bars = broker.fetch_4h_bars(SYMBOLS)
    htf_bars = broker.fetch_daily_bars(SYMBOLS, lookback_days=500)  # >=200 daily bars for the VWAP filter's warmup

    for symbol in SYMBOLS:
        if symbol not in ltf_bars or symbol not in htf_bars:
            continue
        ltf, htf = ltf_bars[symbol], htf_bars[symbol]
        if len(ltf) < 150 or len(htf) < 210:
            continue
        prepared = strat_mtf.prepare(ltf, htf, trend_filter_df=htf, **MTF_PARAMS)
        row = prepared.iloc[-1]
        ts_str = str(prepared.index[-1])
        cursor = state["cursors"].setdefault(symbol, {})
        key = f"{symbol}|mtf"

        if key in state["positions"]:
            s = state["positions"][key]
            # MTF tracks its extreme via Close, not High, and has no activation gate (matching backtest_mtf.py exactly)
            update_trail(symbol, "mtf", s, row, float(row["Close"]), strat_mtf.TRAIL_ATR_MULT, None, args)

        if cursor.get("last_4h_ts") == ts_str:
            continue  # no new 4H bar since last run
        cursor["last_4h_ts"] = ts_str

        if key not in state["positions"] and bool(row["long_entry"]):
            atr_sig = float(row["atr"])
            initial_stop = float(row["Close"]) - strat_mtf.TRAIL_ATR_MULT * atr_sig  # trail active from bar 1
            print(f"  {symbol}: MTF entry signal")
            if args.dry_run:
                continue
            entry = place_entry("mtf", symbol, float(row["Close"]), initial_stop, atr_sig, equity)
            if entry:
                state["positions"][key] = entry


def run_pcse(state, equity, args):
    """PCSE (Probabilistic Candle-State Edge), 4H, long-only, SL=12x entry-time
    return-stdev (fixed) / TP=touch of the 20-bar Bollinger upper band at
    3.0std (moving target). Unlike breakout/mtf's single resting stop, the
    take-profit here is checked explicitly each run rather than kept as a
    second resting order -- avoids ever having two exit orders simultaneously
    committing the same shares. This means TP precision is bounded by this
    runner's check cadence (~every 2h), not truly intrabar like the backtest;
    acceptable given the exit is deliberately wide/patient (FINDINGS_CANDLE_PROB.md),
    not a tight/reactive design this would meaningfully degrade.

    Each run fetches exactly PCSE_BARS_NEEDED (TRAIN_BARS+TEST_BARS) 4H bars
    and fits ONE walk-forward fold on them -- methodologically identical to
    any single fold in the validated backtest (each fold trains independently
    on only its own preceding window, no cross-fold pooling), so this isn't a
    reduced-rigor shortcut, just the minimum computation needed to guarantee
    the latest bar always gets a valid (non-NaN) prediction.
    """
    print("\n--- PCSE (4H, SL=12x / TP=BB(3.0std)) ---")
    bars = broker.fetch_4h_bars(PCSE_SYMBOLS, lookback_days=PCSE_LOOKBACK_DAYS)

    for symbol in PCSE_SYMBOLS:
        if symbol not in bars:
            continue
        df = bars[symbol]
        if len(df) < PCSE_BARS_NEEDED + 5:
            continue
        df = df.tail(PCSE_BARS_NEEDED)  # exact fold alignment -- guarantees the last row is covered

        try:
            prepared = strat_pcse.prepare(df)
        except Exception as e:
            print(f"  {symbol}/pcse: prepare failed: {e}")
            continue
        row = prepared.iloc[-1]
        ts_str = str(prepared.index[-1])
        cursor = state["cursors"].setdefault(symbol, {})
        key = f"{symbol}|pcse"

        # manage existing position: has the moving take-profit been touched?
        if key in state["positions"]:
            s = state["positions"][key]
            bb_mid, bb_std = row.get("bb_mid"), row.get("bb_std")
            if pd.notna(bb_mid) and pd.notna(bb_std):
                tp_price = float(bb_mid) + strat_pcse.BB_NUM_STD * float(bb_std)
                if tp_price > s["entry_price"] and float(row["High"]) >= tp_price:
                    print(f"  {symbol}/pcse: take-profit touched (~{tp_price:.2f}) -- closing")
                    if not args.dry_run:
                        broker.cancel_order(s["stop_order_id"])
                        broker.submit_market_sell(symbol, s["shares"])
                    del state["positions"][key]
                    continue  # don't also evaluate a fresh entry on this same bar

        if cursor.get("last_4h_ts_pcse") == ts_str:
            continue  # no new 4H bar since last run
        cursor["last_4h_ts_pcse"] = ts_str

        if key not in state["positions"] and bool(row["long_entry"]):
            ret_stdev = float(row["ret_stdev"])
            entry_hint = float(row["Close"])
            initial_stop = entry_hint - strat_pcse.SL_MULT * ret_stdev * entry_hint
            print(f"  {symbol}: PCSE entry signal")
            if args.dry_run:
                continue
            entry = place_entry("pcse", symbol, entry_hint, initial_stop, ret_stdev, equity)
            if entry:
                state["positions"][key] = entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== Live trading run: {datetime.now().isoformat()} {'(DRY RUN)' if args.dry_run else ''} ===")

    state = load_state()
    state.setdefault("positions", {})
    state.setdefault("cursors", {})

    equity = broker.get_equity()
    broker_positions = broker.get_open_positions()
    print(f"Account equity: ${equity:,.2f} | distinct symbols held: {len(broker_positions)} | tracked lots: {len(state['positions'])}")

    # reconcile: check each lot's own stop order, not the symbol's combined
    # broker position (another strategy may hold shares of the same symbol)
    for key in list(state["positions"].keys()):
        s = state["positions"][key]
        if args.dry_run:
            continue
        status = broker.get_order_status(s["stop_order_id"])
        if status["status"] == "filled":
            print(f"  {key}: stopped out @ {status['fill_price']:.2f} -- clearing local state")
            del state["positions"][key]
        elif status["status"] in ("canceled", "expired", "rejected"):
            print(f"  {key}: stop order {status['status']} unexpectedly -- clearing local state, check the account manually")
            del state["positions"][key]

    run_breakout(state, equity, args)
    run_mtf(state, equity, args)
    run_pcse(state, equity, args)

    if not args.dry_run:
        save_state(state)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
