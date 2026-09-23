"""Live/paper trading runner #2: the "master combined" bot from the
project-wide deep-dive audit, on its OWN dedicated Alpaca paper account
(broker_alpaca_v2.py / state_store_v2.py) -- fully isolated from
live_trading.py's account, state file, and open positions.

Three strategies, all independently validated in this project's FINDINGS_*.md
docs (see each run_* function's docstring for its own confirmation numbers):

  1. Breakout Hunter (daily) -- strategy_breakout.py, own two-stage ATR
     trailing stop. Confirmed combined t=2.80 (FINDINGS_BREAKOUT.md).
  2. Structure-Confirmed Breakout + PCSE-style exit (daily) -- entries from
     strategy_swing_breakout.py, but its own two-stage trailing stop is
     REPLACED with the wide-fixed-SL/Bollinger-TP exit from
     FINDINGS_BREAKOUT_PCSE_EXIT.md, originally sl_atr_mult=8 (t=6.66
     combined, +86.7% real shared-ledger return, -8.8% max DD -- see that
     file's "sizing trap" section for why sl=8 beat higher-t-stat, wider-stop
     cells that turned out to be a position-sizing artifact). That result was
     only ever validated on the original 40/47-symbol universe though (40
     hand-picked, currently-thriving mega/large-caps, zero laggards or
     turnarounds -- a real survivorship-bias risk). sweep_pcse_exit_sl.py
     re-ran the full SL grid through this exact 3-strategy combo on a more
     realistic 81-symbol universe (+41 known laggards/cyclicals/turnarounds):
     sl=8 there only returns +98.4% AND trips the circuit breaker 2,297
     times, while sl=18 returns +180.4% with ZERO trips and still holds up
     on the original universe (+396.0%) -- SWING_PCSE_SL_ATR_MULT is now 18,
     not 8, the setting that's robust on both universes rather than overfit
     to the friendlier one. bb_num_std=2.5, not that file's original 3.0:
     sweep_swing_pcse_bb_width.py found 2.5 ties 3.0 on combined t-stat (6.72
     vs 6.71) on ~2x the trade count with the best OOS split in the grid
     (t=3.91) -- safe to trust directly, since bb_num_std only moves the
     take-profit level, not position size, so none of the SL sweep's
     sizing-artifact risk applies here.
  3. PCSE (4H) -- strategy_pcse_final.py, own entries and exit unchanged.
     This project's strongest standalone strategy (t=7.06, FINDINGS_CANDLE_PROB.md).

Deliberately NOT included:
  - Hybrid breakout: its entries are a stricter subset of Breakout Hunter's
    squeeze AND Structure-Confirmed's swing-range conditions -- running it
    alongside both parents would mostly duplicate the same signals against
    the same symbols, not add a diversified edge.
  - MTF v2: never cleared this project's t>2 bar across three independent
    re-tests (see live_trading.py's docstring) -- no reason to trade it here either.
  - Orland EWZ (System 1 / Elder): a separate, much-less-validated project
    already given its own isolated risk budget on the OTHER account
    (live_trading.py) -- not duplicated here.

Account-level risk management (peak-equity circuit breaker, daily loss gate,
rolling-expectancy risk throttle) ported verbatim from live_trading.py --
insurance, not alpha, applies regardless of which strategies are wired in.

Run cadence: Breakout Hunter and the Swing+PCSE-exit strategy only need
checking once per day (daily bars); PCSE needs checking roughly every 4
hours during market hours. Schedule this every ~2 hours during market hours,
same as live_trading.py/railway_runner.py.

Usage:
    python live_trading_v2.py            # live paper run, places real (paper) orders
    python live_trading_v2.py --dry-run  # compute and print signals/sizing only
"""
import argparse
from datetime import datetime

import pandas as pd

import bar_cache_v2
import broker_alpaca_v2 as broker
import strategy_breakout as strat_bo
import strategy_pcse_final as strat_pcse
import strategy_swing_breakout as strat_swing
from basket import US_SYMBOLS
from indicators_core import bollinger_bands
from state_store_v2 import load_state, save_state
from test_mtf_oos_symbols import OOS_US_SYMBOLS

SYMBOLS = US_SYMBOLS + OOS_US_SYMBOLS  # 47-symbol IS/OOS universe both breakout strategies were validated on
# Raised from 0.01 to 0.02 -- see live_trading.py's identical comment for the
# full sweep_risk_levels.py / sweep_risk_levels_realistic.py reasoning (2%
# was the best total return in both the original and survivorship-bias-
# corrected universes, with negligible modeled risk of a serious drawdown;
# returns fall and drawdown risk climbs sharply past 2% in both universes).
RISK_PCT = 0.02

# PCSE was only ever validated on basket.US_SYMBOLS (20 symbols) -- not extended
# here beyond what FINDINGS_CANDLE_PROB.md actually tested.
PCSE_SYMBOLS = US_SYMBOLS
PCSE_BARS_NEEDED = strat_pcse.entry_strat.TRAIN_BARS + strat_pcse.entry_strat.TEST_BARS
PCSE_LOOKBACK_DAYS = 800

# Structure-Confirmed + PCSE-exit combo (FINDINGS_BREAKOUT_PCSE_EXIT.md).
# bb_num_std=2.5 (not PCSE's own default of 3.0): sweep_swing_pcse_bb_width.py
# found 2.5 ties 3.0 on combined t-stat (6.72 vs 6.71) on ~2x the trade count
# (1005 vs 522) with the best OOS split in the grid (t=3.91 vs 3.49). Unlike
# the SL sweep, this carries no sizing-trap risk -- bb_num_std only moves the
# take-profit level, not position size, so the t-stat gain here is real.
#
# SL raised from 8 to 18: FINDINGS_BREAKOUT_PCSE_EXIT.md's sl=8 pick was
# validated only on the original 40/47-symbol universe (basket.US_SYMBOLS +
# OOS_US_SYMBOLS) -- 40 hand-picked, currently-thriving mega/large-caps with
# zero laggards or turnaround stories, a real survivorship-bias risk.
# sweep_pcse_exit_sl.py re-ran the full sl_atr_mult grid through the actual
# 3-strategy live_trading_v2.py combo (Breakout Hunter + this + PCSE) on a
# more realistic 81-symbol universe (+41 known laggards/cyclicals/turnarounds,
# see sweep_risk_levels_realistic.py) at today's live RISK_PCT=0.02: sl=8
# there returns only +98.4% AND trips the circuit breaker 2,297 times (the
# strategy keeps slamming into its own drawdown ceiling), while sl=18 returns
# +180.4% with ZERO circuit breaker trips -- and still holds up reasonably on
# the original universe too (+396.0%, vs sl=8's +494.4%). sl=18 is the
# setting that's good on BOTH universes, not just the one it was originally
# tuned on -- the textbook signature of the more robust (less overfit) choice.
SWING_PCSE_SL_ATR_MULT = 18.0
SWING_PCSE_BB_PERIOD = 20
SWING_PCSE_BB_NUM_STD = 2.5

CIRCUIT_BREAKER_DD_PCT = 0.15
DAILY_LOSS_LIMIT_PCT = 0.04
RISK_THROTTLE_MULT = 0.7
RISK_THROTTLE_LOOKBACK = 20
RISK_THROTTLE_CONSEC_LOSSES = 3
PARTIAL_PROFIT_R = 1.2
PARTIAL_PROFIT_FRACTION = 0.3


def last_complete_daily_row(prepared: pd.DataFrame, now=None):
    """Alpaca's historical bars endpoint returns a live, continuously-updating
    PARTIAL bar for whatever calendar day is currently in progress -- confirmed
    empirically (BTC/USD's "today" daily bar already appears mid-day with volume
    far below a completed day's; crypto trades 24/7 so this is directly
    observable any time, unlike equities which only reveal it during market
    hours). Daily bars are timestamped at midnight UTC of the trading day, not
    at market close, so a simple date-equality check -- not a +1-day arithmetic
    check -- is the right way to detect "the last bar is still today's."

    None of this bot's scheduled run times (9:35am-3:35pm ET) land after market
    close, so `prepared.iloc[-1]` is routinely still-forming data. Using it to
    decide an ENTRY would fire off a fraction of a day's move (as little as 5
    minutes' worth on the first run of the day), not the completed close every
    strategy here was backtested against. Returns the last bar whose trading
    day has actually finished, for ENTRY decisions only -- exit monitoring
    (trailing stops, take-profit touches) should keep using `iloc[-1]`
    directly; checking the live/current bar's High is exactly the "intrabar"
    monitoring those exits were designed around, not a bug to fix.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    if prepared.index[-1].date() == now.date():
        return prepared.iloc[-2], prepared.index[-2]
    return prepared.iloc[-1], prepared.index[-1]


def last_complete_bar_by_duration(prepared: pd.DataFrame, bar_duration: pd.Timedelta, now=None):
    """Same issue as last_complete_daily_row, for intraday bars (PCSE's 4H)
    whose timestamps are real clock times, not midnight-shifted -- so a
    straightforward "has this bar's period actually elapsed" check applies."""
    now = now or pd.Timestamp.now(tz="UTC")
    if now < prepared.index[-1] + bar_duration:
        return prepared.iloc[-2], prepared.index[-2]
    return prepared.iloc[-1], prepared.index[-1]


def place_entry(strategy, symbol, qty_hint_price, initial_stop, atr_at_signal, equity, risk_pct):
    stop_distance = qty_hint_price - initial_stop
    if stop_distance <= 0:
        return None
    risk_amount = equity * risk_pct
    # Cap by buying power, not just equity -- see broker_alpaca.py's
    # get_buying_power() / submit_market_buy() for why equity alone
    # was producing orders Alpaca rejected with "insufficient buying power".
    buying_power = broker.get_buying_power()
    qty = int(min(risk_amount / stop_distance, equity / qty_hint_price, buying_power / qty_hint_price))
    if qty < 1:
        return None

    fill = broker.submit_market_buy(symbol, qty)
    if not fill["filled"]:
        if fill.get("rejected"):
            print(f"    {symbol}/{strategy}: order rejected -- {fill['error']}")
        else:
            print(f"    {symbol}/{strategy}: order not filled within wait window, skipping")
        return None

    entry_price = fill["fill_price"]
    filled_qty = fill["qty"]
    stop_order_id = broker.submit_stop_sell(symbol, filled_qty, initial_stop)
    print(f"    {symbol}/{strategy}: filled @ {entry_price:.2f} qty={filled_qty} -- stop @ {initial_stop:.2f}")

    return {
        "entry_price": entry_price, "shares": filled_qty, "atr_at_entry": atr_at_signal,
        "current_stop": initial_stop, "initial_stop": initial_stop, "trail_active": False,
        "extreme": entry_price, "stop_order_id": stop_order_id, "partial_taken": False,
    }


def update_trail(symbol, strategy, s, row, extreme_price, trail_mult, activate_atr, args):
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
        current_price = float(row["Close"])
        if new_stop > s["current_stop"]:
            if new_stop >= current_price:
                # extreme was set on an earlier, higher bar; a sharp pullback
                # since then can make the trail formula compute a stop AT or
                # ABOVE the current price, which Alpaca rejects outright (a
                # sell-stop can't sit above market). Cancelling the old
                # (still valid) resting stop before that submission failed
                # would leave the position with NO stop order at all --
                # skip the raise instead, leaving the existing valid stop
                # order in place untouched.
                print(f"  {symbol}/{strategy}: skipping stop raise to {new_stop:.2f} -- at/above current price {current_price:.2f}")
            else:
                print(f"  {symbol}/{strategy}: raising stop {s['current_stop']:.2f} -> {new_stop:.2f}")
                if not args.dry_run:
                    # Submitting the new stop BEFORE cancelling the old one fails
                    # outright (Alpaca: "insufficient qty available", since the
                    # resting old stop already reserves these shares) -- cancel
                    # must go first. But that means ANY failure submitting the
                    # replacement (not just the invalid-price case above -- a
                    # transient API error is enough) leaves the position with NO
                    # stop at all. This happened live: AMD's raise cancelled a
                    # valid stop, the resubmit failed for an unlogged reason, and
                    # the position sat unprotected until reconciliation cleared it
                    # next run and Breakout Hunter opened a SECOND position on top
                    # of the still-held, now-untracked first one. Re-establish the
                    # OLD (already-valid) price on any failure instead of leaving
                    # the cancel to stand alone.
                    old_stop_price = s["current_stop"]
                    broker.cancel_order(s["stop_order_id"])
                    try:
                        s["stop_order_id"] = broker.submit_stop_sell(symbol, pos_qty, new_stop)
                        s["current_stop"] = new_stop
                    except Exception as e:
                        print(f"  {symbol}/{strategy}: raised-stop submission failed ({e!r}) -- "
                              f"re-establishing the old stop ({old_stop_price:.2f}) instead of leaving the position unprotected")
                        s["stop_order_id"] = broker.submit_stop_sell(symbol, pos_qty, old_stop_price)
                else:
                    s["current_stop"] = new_stop


def check_partial_profit(symbol, strategy, s, current_price, args):
    if s.get("partial_taken") or s.get("initial_stop") is None:
        return
    r_distance = s["entry_price"] - s["initial_stop"]
    if r_distance <= 0:
        return
    favorable_r = (current_price - s["entry_price"]) / r_distance
    if favorable_r < PARTIAL_PROFIT_R:
        return
    qty_to_sell = int(s["shares"] * PARTIAL_PROFIT_FRACTION)
    if qty_to_sell < 1:
        return
    print(f"  {symbol}/{strategy}: +{favorable_r:.2f}R -- taking partial profit on {qty_to_sell}/{s['shares']} shares")
    if args.dry_run:
        return
    broker.cancel_order(s["stop_order_id"])
    broker.submit_market_sell(symbol, qty_to_sell)
    s["shares"] -= qty_to_sell
    s["stop_order_id"] = broker.submit_stop_sell(symbol, s["shares"], s["current_stop"])
    s["partial_taken"] = True


def record_closed_trade(state, s, exit_price):
    initial_stop = s.get("initial_stop")
    if initial_stop is None:
        return
    r_distance = s["entry_price"] - initial_stop
    if r_distance <= 0:
        return
    r_multiple = (exit_price - s["entry_price"]) / r_distance
    log = state.setdefault("trade_log", [])
    log.append({"r": r_multiple, "win": r_multiple > 0})
    del log[:-200]


def update_risk_throttle(state, equity, peak_equity):
    log = state.get("trade_log", [])
    three_losses = len(log) >= RISK_THROTTLE_CONSEC_LOSSES and all(
        not t["win"] for t in log[-RISK_THROTTLE_CONSEC_LOSSES:]
    )
    neg_expectancy = len(log) >= RISK_THROTTLE_LOOKBACK and (
        sum(t["r"] for t in log[-RISK_THROTTLE_LOOKBACK:]) / RISK_THROTTLE_LOOKBACK < 0
    )
    if three_losses or neg_expectancy:
        state["risk_throttled"] = True
    if equity >= peak_equity:
        state["risk_throttled"] = False
    return RISK_PCT * RISK_THROTTLE_MULT if state.get("risk_throttled") else RISK_PCT


def run_breakout(state, equity, risk_pct, halt_entries, args, bars):
    print("\n--- Breakout Hunter (daily) ---")

    for symbol, df in bars.items():
        if len(df) < strat_bo.PERCENTILE_LOOKBACK + 50:
            continue
        prepared = strat_bo.prepare(df)
        row = prepared.iloc[-1]  # exit management: today's live-updating bar, correct for intrabar monitoring
        cursor = state["cursors"].setdefault(symbol, {})
        key = f"{symbol}|breakout"

        if key in state["positions"]:
            s = state["positions"][key]
            check_partial_profit(symbol, "breakout", s, float(row["Close"]), args)
            update_trail(symbol, "breakout", s, row, float(row["High"]), strat_bo.TRAIL_ATR_MULT, strat_bo.TRAIL_ACTIVATE_ATR, args)

        # entry decision: last COMPLETED bar, not today's still-forming one -- see last_complete_daily_row
        entry_row, entry_ts = last_complete_daily_row(prepared)
        entry_date_str = str(entry_ts.date())
        if cursor.get("last_daily_date") == entry_date_str:
            continue
        cursor["last_daily_date"] = entry_date_str

        if key not in state["positions"] and bool(entry_row["long_entry"]):
            support, atr_sig = float(entry_row["support"]), float(entry_row["atr"])
            initial_stop = support - strat_bo.INITIAL_STOP_ATR_BUFFER * atr_sig
            print(f"  {symbol}: BREAKOUT entry signal" + (" (entries halted)" if halt_entries else ""))
            if args.dry_run or halt_entries:
                continue
            entry = place_entry("breakout", symbol, float(entry_row["Close"]), initial_stop, atr_sig, equity, risk_pct)
            if entry:
                state["positions"][key] = entry


def run_swing_pcse_exit(state, equity, risk_pct, halt_entries, args, bars):
    """Structure-Confirmed Breakout entries (strategy_swing_breakout.py),
    PCSE-style exit: FIXED stop-loss at entry_price - SWING_PCSE_SL_ATR_MULT x
    ATR-at-entry (never trails, unlike run_breakout's two-stage stop), take-profit
    at the first touch of the 20-bar Bollinger upper band at SWING_PCSE_BB_NUM_STD
    std devs (a moving target, re-checked every run). See FINDINGS_BREAKOUT_PCSE_EXIT.md
    for the original validation (t=6.66 combined, +86.7% real shared-ledger return,
    -8.8% max DD) and why that file picked sl=8 over higher-t-stat wider-stop cells
    (a sizing artifact) -- SWING_PCSE_SL_ATR_MULT is now 18, not that file's 8, per
    sweep_pcse_exit_sl.py's realistic-universe re-test; see this module's own
    docstring (strategy #2) for the full reasoning.

    Same "checked periodically, not truly intrabar" caveat as run_pcse's own
    take-profit check -- acceptable given this is a deliberately wide/patient exit.
    No partial-profit-taking here, matching PCSE's own exit (deliberately left
    unmodified from its validated form).
    """
    print(f"\n--- Structure-Confirmed Breakout + PCSE-exit (daily, SL={SWING_PCSE_SL_ATR_MULT}x/TP=BB({SWING_PCSE_BB_NUM_STD}std)) ---")

    for symbol, df in bars.items():
        if len(df) < 2 * strat_swing.SWING_LEFT + 2 * strat_swing.SWING_RIGHT + 50:
            continue
        prepared = strat_swing.prepare(df)
        bb_mid, bb_upper, _ = bollinger_bands(prepared["Close"], SWING_PCSE_BB_PERIOD, SWING_PCSE_BB_NUM_STD)
        prepared["bb_upper"] = bb_upper
        row = prepared.iloc[-1]  # exit management: today's live-updating bar, correct for intrabar monitoring
        cursor = state["cursors"].setdefault(symbol, {})
        key = f"{symbol}|swing_pcse"

        if key in state["positions"]:
            s = state["positions"][key]
            tp_price = row["bb_upper"]
            if pd.notna(tp_price) and tp_price > s["entry_price"] and float(row["High"]) >= tp_price:
                print(f"  {symbol}/swing_pcse: take-profit touched (~{tp_price:.2f}) -- closing")
                if not args.dry_run:
                    broker.cancel_order(s["stop_order_id"])
                    fill = broker.submit_market_sell(symbol, s["shares"])
                    exit_price = fill["fill_price"] if fill.get("filled") else float(tp_price)
                    record_closed_trade(state, s, exit_price)
                del state["positions"][key]
                continue

        # entry decision: last COMPLETED bar, not today's still-forming one -- see last_complete_daily_row
        entry_row, entry_ts = last_complete_daily_row(prepared)
        entry_date_str = str(entry_ts.date())
        if cursor.get("last_daily_date_swing_pcse") == entry_date_str:
            continue
        cursor["last_daily_date_swing_pcse"] = entry_date_str

        if key not in state["positions"] and bool(entry_row["long_entry"]):
            atr_sig = float(entry_row["atr"])
            entry_hint = float(entry_row["Close"])
            initial_stop = entry_hint - SWING_PCSE_SL_ATR_MULT * atr_sig
            print(f"  {symbol}: SWING+PCSE-EXIT entry signal" + (" (entries halted)" if halt_entries else ""))
            if args.dry_run or halt_entries:
                continue
            entry = place_entry("swing_pcse", symbol, entry_hint, initial_stop, atr_sig, equity, risk_pct)
            if entry:
                state["positions"][key] = entry


def run_pcse(state, equity, risk_pct, halt_entries, args):
    print("\n--- PCSE (4H, SL=12x / TP=BB(3.0std)) ---")
    bars = bar_cache_v2.fetch_4h_bars_cached(PCSE_SYMBOLS, lookback_days=PCSE_LOOKBACK_DAYS)

    for symbol in PCSE_SYMBOLS:
        if symbol not in bars:
            continue
        df = bars[symbol]
        if len(df) < PCSE_BARS_NEEDED + 5:
            continue
        df = df.tail(PCSE_BARS_NEEDED)

        try:
            prepared = strat_pcse.prepare(df)
        except Exception as e:
            print(f"  {symbol}/pcse: prepare failed: {e}")
            continue
        row = prepared.iloc[-1]  # exit management: current live-updating 4H bar, correct for intrabar monitoring
        cursor = state["cursors"].setdefault(symbol, {})
        key = f"{symbol}|pcse"

        if key in state["positions"]:
            s = state["positions"][key]
            bb_mid, bb_std = row.get("bb_mid"), row.get("bb_std")
            if pd.notna(bb_mid) and pd.notna(bb_std):
                tp_price = float(bb_mid) + strat_pcse.BB_NUM_STD * float(bb_std)
                if tp_price > s["entry_price"] and float(row["High"]) >= tp_price:
                    print(f"  {symbol}/pcse: take-profit touched (~{tp_price:.2f}) -- closing")
                    if not args.dry_run:
                        broker.cancel_order(s["stop_order_id"])
                        fill = broker.submit_market_sell(symbol, s["shares"])
                        exit_price = fill["fill_price"] if fill.get("filled") else tp_price
                        record_closed_trade(state, s, exit_price)
                    del state["positions"][key]
                    continue

        # entry decision: last COMPLETED 4H bar, not the currently-forming one -- see last_complete_bar_by_duration
        entry_row, entry_ts = last_complete_bar_by_duration(prepared, pd.Timedelta(hours=4))
        ts_str = str(entry_ts)
        if cursor.get("last_4h_ts_pcse") == ts_str:
            continue
        cursor["last_4h_ts_pcse"] = ts_str

        if key not in state["positions"] and bool(entry_row["long_entry"]):
            ret_stdev = float(entry_row["ret_stdev"])
            entry_hint = float(entry_row["Close"])
            initial_stop = entry_hint - strat_pcse.SL_MULT * ret_stdev * entry_hint
            print(f"  {symbol}: PCSE entry signal" + (" (entries halted)" if halt_entries else ""))
            if args.dry_run or halt_entries:
                continue
            entry = place_entry("pcse", symbol, entry_hint, initial_stop, ret_stdev, equity, risk_pct)
            if entry:
                state["positions"][key] = entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== live_trading_v2 run: {datetime.now().isoformat()} {'(DRY RUN)' if args.dry_run else ''} ===")

    state = load_state()
    state.setdefault("positions", {})
    state.setdefault("cursors", {})
    state.setdefault("trade_log", [])

    equity = broker.get_equity()
    broker_positions = broker.get_open_positions()
    print(f"Account equity: ${equity:,.2f} | distinct symbols held: {len(broker_positions)} | tracked lots: {len(state['positions'])}")

    for key in list(state["positions"].keys()):
        s = state["positions"][key]
        if args.dry_run:
            continue
        status = broker.get_order_status(s["stop_order_id"])
        if status["status"] == "filled":
            print(f"  {key}: stopped out @ {status['fill_price']:.2f} -- clearing local state")
            record_closed_trade(state, s, status["fill_price"])
            del state["positions"][key]
        elif status["status"] in ("canceled", "expired", "rejected"):
            print(f"  {key}: stop order {status['status']} unexpectedly -- clearing local state, check the account manually")
            del state["positions"][key]

    peak_equity = max(state.get("peak_equity", equity), equity)
    state["peak_equity"] = peak_equity
    drawdown_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

    today_str = str(datetime.now().date())
    if state.get("daily_start_date") != today_str:
        state["daily_start_date"] = today_str
        state["daily_start_equity"] = equity
    daily_start_equity = state.get("daily_start_equity", equity)
    daily_loss_pct = (daily_start_equity - equity) / daily_start_equity if daily_start_equity > 0 else 0.0

    halt_entries = False
    if drawdown_pct >= CIRCUIT_BREAKER_DD_PCT:
        print(f"\n!!! CIRCUIT BREAKER: equity ${equity:,.2f} is {drawdown_pct:.1%} below peak ${peak_equity:,.2f} -- new entries halted")
        halt_entries = True
    elif daily_loss_pct >= DAILY_LOSS_LIMIT_PCT:
        print(f"\n!!! DAILY LOSS GATE: equity down {daily_loss_pct:.1%} today (from ${daily_start_equity:,.2f}) -- new entries halted until tomorrow")
        halt_entries = True

    risk_pct = update_risk_throttle(state, equity, peak_equity)
    if risk_pct < RISK_PCT:
        print(f"  risk throttled to {risk_pct:.2%} of equity per trade (recent performance below bar)")

    daily_bars = broker.fetch_daily_bars(SYMBOLS)  # fetched once, shared by both daily strategies below
    # try/finally: one strategy raising shouldn't skip save_state for
    # whatever the others already did this pass -- see live_trading.py's
    # identical fix for the full reasoning (an unsaved fill re-signals next
    # run and collides with its own still-open broker-side stop order).
    try:
        run_breakout(state, equity, risk_pct, halt_entries, args, daily_bars)
        run_swing_pcse_exit(state, equity, risk_pct, halt_entries, args, daily_bars)
        run_pcse(state, equity, risk_pct, halt_entries, args)
    except Exception as e:
        print(f"\n!!! run aborted by unexpected error: {e!r} -- saving state for whatever completed above")
    finally:
        if not args.dry_run:
            save_state(state)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
