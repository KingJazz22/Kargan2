"""Unified live/paper trading runner: Breakout Hunter (daily) + PCSE (4H,
SL=12x/TP=BB(3.0std)) + Swing-Structure Breakout (daily, structure-confirmed)
sharing ONE Alpaca paper account -- the three validated strategies from this
project (Grid-Martingale deliberately excluded -- its own
FINDINGS_GRID_MARTINGALE.md says "still not ready for live paper trading";
the swing-structure pullback variant and every other researched strategy --
Channel MACD, options spreads, mean reversion, the 1m-anchored
MTF3/RSI/Stoch-threshold sweeps -- are excluded because their own FINDINGS_*.md
either found no real edge or explicitly flagged the result as too preliminary
to trade).

Multi-Timeframe RSI+Stochastic v2 (4H execution + Daily confirm, TSL-only
exit, VWAP(200) trend filter) is wired in (`run_mtf` below) but its new
entries are DISABLED via MTF_ENTRIES_DISABLED -- see that flag's comment.
Three independent re-tests (original 10.5yr Alpaca deep history; this
project's shared-capital-pool combined-system engine on the same data;
a vectorbt/yfinance-sourced re-run after the project-wide gap-through-fill
bug fix) have now all found the SAME thing: t stays in the +1.0 to +1.9
range, never clearing this project's own t>2 significance bar, across every
data source and window tried. The code path is kept live (not deleted) only
so it keeps managing any already-open MTF position via the same
halt-new-entries pattern the circuit breaker uses -- there are currently none.

Swing-Structure Breakout (`strategy_swing_breakout.py`) is treated with full
parity to Breakout Hunter/MTF/PCSE (same risk_pct, same circuit
breaker/throttle/partial-profit machinery) rather than Orland's reduced-risk
side-account treatment -- FINDINGS_SWING_STRUCTURE.md's own verdict is "treat
it the way Breakout Hunter is treated," and its validation (in-sample t=4.41,
out-of-sample t≈3.0, OOS independently significant on its own) is stronger
than Breakout Hunter's own (OOS came back flat there). Same US-equities-only
universe as Breakout Hunter (SYMBOLS) -- FINDINGS_BREAKOUT.md found no crypto
edge, and this strategy's own validation never tested crypto either.

Account-level risk management (added after scavenging two other Railway bots
before retiring them -- see commit history): a peak-equity circuit breaker, a
daily-loss gate, a rolling-expectancy risk throttle, and partial profit-taking
on breakout/mtf. These are execution/risk-discipline patterns borrowed from
systems that ran them live, NOT independently backtested against Kargan2's
own strategies -- unlike everything else in this file, there's no
FINDINGS_*.md validating them against this project's edges. Treat them as
insurance, not alpha. PCSE's exit logic is deliberately left untouched (its
fixed-SL/moving-TP design is already validated by FINDINGS_CANDLE_PROB.md;
bolting on an unvalidated partial-exit would compromise that). Also
deliberately NOT ported: the "stale limit-exit order" fix from ALG_TRADINGV2
-- this file uses real STOP orders (trigger-to-market), not resting limit
exits, so that specific failure mode doesn't apply here.

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
from indicators_core import atr

import broker_alpaca as broker
import strategy_breakout as strat_bo
import strategy_mtf as strat_mtf
import strategy_pcse_final as strat_pcse
import strategy_swing_breakout as strat_swing
from basket import US_SYMBOLS
from orland_ewz import data_fetch_alpaca as orland_data
from orland_ewz import elder_exits, elder_strategy
from orland_ewz import levels as orland_levels
from orland_ewz import mtf_score as orland_mtf_score
from orland_ewz import strategy_s1
from state_store import load_state, save_state
from test_mtf_oos_symbols import OOS_US_SYMBOLS

SYMBOLS = US_SYMBOLS + OOS_US_SYMBOLS  # the 40 symbols both edges were validated on
# Raised from 0.01 to 0.02: sweep_risk_levels.py / sweep_risk_levels_realistic.py
# swept 0.5%-5% risk/trade through the real shared-ledger engine (circuit
# breaker, daily loss gate, cash constraints) on both the original 40-symbol
# universe and a more realistic 81-symbol one (adds known laggards/turnarounds
# to counter survivorship bias) -- 2% came out as the best total return in
# BOTH universes and both the historical replay and Monte Carlo layers, with
# negligible modeled risk of a serious drawdown (0.1% chance of a 30%+ DD on
# the realistic universe). Past 2%, returns fall and drawdown risk climbs
# sharply in both universes -- this is not "more risk, more reward" without
# limit, 2% is the peak, not a floor.
RISK_PCT = 0.02
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

# MTF v2 has never cleared this project's t>2 significance bar across three
# independent re-tests (see module docstring). Halts NEW entries only --
# `run_mtf` keeps running so it still manages any already-open MTF position
# (there are none as of the date this flag was added), same pattern as the
# circuit breaker's halt_entries. Flip back to False if a future re-test
# clears the bar.
MTF_ENTRIES_DISABLED = True

# Circuit breaker: halt NEW entries (existing positions still get managed --
# trails still raised, PCSE take-profits still taken) once equity draws down
# this much from its running peak. New entries resume once equity recovers.
CIRCUIT_BREAKER_DD_PCT = 0.15
# Daily loss gate: halt new entries for the rest of the calendar day once
# equity is down this much from today's first-run equity.
DAILY_LOSS_LIMIT_PCT = 0.04
# Rolling-expectancy risk throttle: cut position-sizing risk once recent
# closed-trade history turns negative, restore once equity makes a new peak.
RISK_THROTTLE_MULT = 0.7
RISK_THROTTLE_LOOKBACK = 20
RISK_THROTTLE_CONSEC_LOSSES = 3
# Partial profit-taking (breakout/mtf only, see module docstring): once a
# position is this many R favorable, bank part of it and let the rest ride
# the existing trailing stop.
PARTIAL_PROFIT_R = 1.2
PARTIAL_PROFIT_FRACTION = 0.3

# --- OrlandMagics EWZ integration -------------------------------------------
# Two strategies ported from a separate, much-less-validated project
# (OrlandMagics) -- neither clears this project's usual bar (t>=3.40,
# in/out-of-sample, walk-forward, Monte Carlo). Wired in anyway at the user's
# explicit request, but given their own account/circuit breaker rather than
# full parity with breakout/mtf/pcse, so a bad run here can't eat into the
# budget the vetted strategies depend on -- see orland_ewz/ for the ported
# modules and the plan/memory notes for the full validation-gap discussion.
ORLAND_SYMBOL = "EWZ"
ORLAND_RISK_PCT = RISK_PCT * 0.5  # half the vetted strategies' per-trade risk
# Halt NEW orland entries (existing orland positions still get managed) after
# this many closed EWZ losses in a row, tracked in state["orland_trade_log"]
# -- completely separate from the main risk_throttle/trade_log above. Orland
# losses can never trip the breaker that protects breakout/mtf/pcse's
# budget, and vice versa; the shared account-level circuit breaker/daily-loss
# gate still applies to everyone, orland included (see main()).
ORLAND_CONSEC_LOSS_HALT = 3


def place_entry(strategy, symbol, qty_hint_price, initial_stop, atr_at_signal, equity, risk_pct, extra=None):
    """Submits the market buy, waits for fill, places the protective stop,
    and returns the new state entry (or None if not filled/sizeable)."""
    stop_distance = qty_hint_price - initial_stop
    if stop_distance <= 0:
        return None
    risk_amount = equity * risk_pct
    # Cap by *buying power*, not equity -- equity includes non-marginable
    # positions (e.g. crypto, which ties up cash 1:1 with no margin) and
    # already-committed margin, neither of which is actually spendable. Sizing
    # off equity alone was producing orders Alpaca then rejected outright with
    # "insufficient buying power", losing the whole run (see submit_market_buy).
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

    entry = {
        "entry_price": entry_price, "shares": filled_qty, "atr_at_entry": atr_at_signal,
        "current_stop": initial_stop, "initial_stop": initial_stop, "trail_active": False,
        "extreme": entry_price, "stop_order_id": stop_order_id, "partial_taken": False,
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
        current_price = float(row["Close"])
        if new_stop > s["current_stop"]:
            if new_stop >= current_price:
                # extreme was set on an earlier, higher bar; a sharp pullback
                # since then can make the trail formula compute a stop AT or
                # ABOVE the current price, which Alpaca rejects outright (a
                # sell-stop can't sit above market). Cancelling the old
                # (still valid) resting stop before that submission failed is
                # exactly how a CRM lot ended up with a live position and
                # NO stop order at all -- skip the raise instead, leaving the
                # existing valid stop order in place untouched.
                print(f"  {symbol}/{strategy}: skipping stop raise to {new_stop:.2f} -- at/above current price {current_price:.2f}")
            else:
                print(f"  {symbol}/{strategy}: raising stop {s['current_stop']:.2f} -> {new_stop:.2f}")
                if not args.dry_run:
                    broker.cancel_order(s["stop_order_id"])
                    s["stop_order_id"] = broker.submit_stop_sell(symbol, pos_qty, new_stop)
                s["current_stop"] = new_stop


def check_partial_profit(symbol, strategy, s, current_price, args):
    """Banks PARTIAL_PROFIT_FRACTION of the position the first time it
    reaches PARTIAL_PROFIT_R favorable R-multiple, then re-rests the stop
    for the reduced quantity. One-shot per lot (s["partial_taken"]) --
    doesn't repeat on later bars. Skips lots opened before this field
    existed (no initial_stop -> no R-multiple to measure against)."""
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


def record_closed_trade(state, s, exit_price, log_key="trade_log"):
    """Logs an R-multiple outcome for the rolling-expectancy risk throttle
    (or, for orland strategies, their own separate consecutive-loss halt --
    see log_key). Only full closes are logged (not partial profit-takes) to
    avoid double-counting a single position's outcome."""
    initial_stop = s.get("initial_stop")
    if initial_stop is None:
        return  # pre-existing lot from before this field existed
    r_distance = s["entry_price"] - initial_stop
    if r_distance <= 0:
        return
    r_multiple = (exit_price - s["entry_price"]) / r_distance
    log = state.setdefault(log_key, [])
    log.append({"r": r_multiple, "win": r_multiple > 0})
    del log[:-200]  # keep bounded; no-op while the log is still short


def orland_entries_halted(state) -> bool:
    """True once the last ORLAND_CONSEC_LOSS_HALT closed orland trades were
    all losses. Self-clearing: the moment a win lands in that trailing
    window, this goes False again -- no separate reset flag needed, unlike
    the main risk_throttle's peak-equity reset (this is a hard halt on new
    entries, not a size cut, so it doesn't need the same restore condition)."""
    log = state.get("orland_trade_log", [])
    return len(log) >= ORLAND_CONSEC_LOSS_HALT and all(
        not t["win"] for t in log[-ORLAND_CONSEC_LOSS_HALT:]
    )


def update_risk_throttle(state, equity, peak_equity):
    """Cuts risk_pct once recent closed-trade history turns negative
    (3-in-a-row losses, or negative expectancy over the last
    RISK_THROTTLE_LOOKBACK trades), restores it once equity makes a new
    peak. Returns the risk_pct to use for sizing THIS run's new entries."""
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


def run_breakout(state, equity, risk_pct, halt_entries, args):
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
            check_partial_profit(symbol, "breakout", s, float(row["Close"]), args)
            update_trail(symbol, "breakout", s, row, float(row["High"]), strat_bo.TRAIL_ATR_MULT, strat_bo.TRAIL_ACTIVATE_ATR, args)

        if cursor.get("last_daily_date") == today_str:
            continue  # already processed today's bar
        cursor["last_daily_date"] = today_str

        if key not in state["positions"] and bool(row["long_entry"]):
            support, atr_sig = float(row["support"]), float(row["atr"])
            initial_stop = support - strat_bo.INITIAL_STOP_ATR_BUFFER * atr_sig
            print(f"  {symbol}: BREAKOUT entry signal" + (" (entries halted)" if halt_entries else ""))
            if args.dry_run or halt_entries:
                continue
            entry = place_entry("breakout", symbol, float(row["Close"]), initial_stop, atr_sig, equity, risk_pct)
            if entry:
                state["positions"][key] = entry


def run_swing_breakout(state, equity, risk_pct, halt_entries, args):
    """Structure-confirmed breakout (FINDINGS_SWING_STRUCTURE.md), same daily
    cadence and two-stage stop shape as run_breakout -- deliberately uses its
    own cursor key ("last_daily_date_swing") in the shared per-symbol cursor
    dict rather than breakout's "last_daily_date", since main() runs both in
    the same pass and sharing a key would make this strategy see "already
    processed today" on its very first check of the day."""
    print("\n--- Swing-Structure Breakout (daily, structure-confirmed) ---")
    bars = broker.fetch_daily_bars(SYMBOLS)

    for symbol, df in bars.items():
        if len(df) < 2 * strat_swing.SWING_LEFT + 2 * strat_swing.SWING_RIGHT + 50:
            continue
        prepared = strat_swing.prepare(df)
        row = prepared.iloc[-1]
        today_str = str(prepared.index[-1].date())
        cursor = state["cursors"].setdefault(symbol, {})
        key = f"{symbol}|swing_breakout"

        # manage existing position (tracks its extreme via High, matching backtest_swing_breakout.py)
        if key in state["positions"]:
            s = state["positions"][key]
            check_partial_profit(symbol, "swing_breakout", s, float(row["Close"]), args)
            update_trail(symbol, "swing_breakout", s, row, float(row["High"]), strat_swing.TRAIL_ATR_MULT, strat_swing.TRAIL_ACTIVATE_ATR, args)

        if cursor.get("last_daily_date_swing") == today_str:
            continue  # already processed today's bar
        cursor["last_daily_date_swing"] = today_str

        if key not in state["positions"] and bool(row["long_entry"]):
            support, atr_sig = float(row["support"]), float(row["atr"])
            initial_stop = support - strat_swing.INITIAL_STOP_ATR_BUFFER * atr_sig
            print(f"  {symbol}: SWING BREAKOUT entry signal" + (" (entries halted)" if halt_entries else ""))
            if args.dry_run or halt_entries:
                continue
            entry = place_entry("swing_breakout", symbol, float(row["Close"]), initial_stop, atr_sig, equity, risk_pct)
            if entry:
                state["positions"][key] = entry


def run_mtf(state, equity, risk_pct, halt_entries, args):
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
            check_partial_profit(symbol, "mtf", s, float(row["Close"]), args)
            update_trail(symbol, "mtf", s, row, float(row["Close"]), strat_mtf.TRAIL_ATR_MULT, None, args)

        if cursor.get("last_4h_ts") == ts_str:
            continue  # no new 4H bar since last run
        cursor["last_4h_ts"] = ts_str

        if key not in state["positions"] and bool(row["long_entry"]):
            atr_sig = float(row["atr"])
            initial_stop = float(row["Close"]) - strat_mtf.TRAIL_ATR_MULT * atr_sig  # trail active from bar 1
            print(f"  {symbol}: MTF entry signal" + (" (entries halted)" if halt_entries else ""))
            if args.dry_run or halt_entries:
                continue
            entry = place_entry("mtf", symbol, float(row["Close"]), initial_stop, atr_sig, equity, risk_pct)
            if entry:
                state["positions"][key] = entry


def run_pcse(state, equity, risk_pct, halt_entries, args):
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
                        fill = broker.submit_market_sell(symbol, s["shares"])
                        exit_price = fill["fill_price"] if fill.get("filled") else tp_price
                        record_closed_trade(state, s, exit_price)
                    del state["positions"][key]
                    continue  # don't also evaluate a fresh entry on this same bar

        if cursor.get("last_4h_ts_pcse") == ts_str:
            continue  # no new 4H bar since last run
        cursor["last_4h_ts_pcse"] = ts_str

        if key not in state["positions"] and bool(row["long_entry"]):
            ret_stdev = float(row["ret_stdev"])
            entry_hint = float(row["Close"])
            initial_stop = entry_hint - strat_pcse.SL_MULT * ret_stdev * entry_hint
            print(f"  {symbol}: PCSE entry signal" + (" (entries halted)" if halt_entries else ""))
            if args.dry_run or halt_entries:
                continue
            entry = place_entry("pcse", symbol, entry_hint, initial_stop, ret_stdev, equity, risk_pct)
            if entry:
                state["positions"][key] = entry


def run_orland_s1(state, equity, halt_entries, args):
    """System 1 (MTF sync + W-formation), ported from OrlandMagics -- see
    orland_ewz/strategy_s1.py's module docstring for what was dropped in the
    port. Long-only, like every other strategy in this file: this project's
    live infra only ever submits BUY-then-SELL-stop orders, so a "down"
    signal from strategy_s1 is simply skipped rather than shorted."""
    print("\n--- Orland System 1 (MTF+W-formation, EWZ) ---")
    timeframe_data = orland_data.get_multi_timeframe(ORLAND_SYMBOL)
    daily_df = timeframe_data.get("1D")
    if daily_df is None or daily_df.empty:
        return
    levels_by_tf = {tf: orland_levels.compute_levels(strategy_s1.trim_for_patterns(df))
                     for tf, df in timeframe_data.items() if df is not None and len(df) > 0}

    key = f"{ORLAND_SYMBOL}|orland_s1"
    cursor = state["cursors"].setdefault(ORLAND_SYMBOL, {})

    if key in state["positions"]:
        s = state["positions"][key]
        position = strategy_s1.Position(
            symbol=ORLAND_SYMBOL, direction=s.get("direction", "up"), entry_price=s["entry_price"],
            stop=s["current_stop"], initial_stop=s["initial_stop"], entry_atr=s["atr_at_entry"],
        )
        position.trail_active = s.get("trail_active", False)
        position.extreme_price = s.get("extreme", s["entry_price"])

        daily_levels = levels_by_tf.get("1D", {})
        atr_val = float(atr(daily_df["High"], daily_df["Low"], daily_df["Close"]).iloc[-1])
        bar = daily_df.iloc[-1]
        regimes = strategy_s1.compute_regimes(timeframe_data)
        mtf = (orland_mtf_score.compute_mtf_score(regimes, symbol=ORLAND_SYMBOL)
               if len(regimes) == len(timeframe_data) else None)
        context = {"atr_val": atr_val, "levels": daily_levels, "mtf": mtf}

        exit_rule = strategy_s1.EXIT_RULES["regime_flip_plus_atr_trail"]
        reason = exit_rule(position, bar, context)

        if reason in ("regime_flip", "consolidation_zone"):
            # Proactive close -- not a price-based stop, so the resting stop
            # order on the broker side wouldn't catch this on its own.
            print(f"  {ORLAND_SYMBOL}/orland_s1: {reason} -- closing")
            if not args.dry_run:
                broker.cancel_order(s["stop_order_id"])
                fill = broker.submit_market_sell(ORLAND_SYMBOL, s["shares"])
                exit_price = fill["fill_price"] if fill.get("filled") else float(bar["Close"])
                record_closed_trade(state, s, exit_price, log_key="orland_trade_log")
            del state["positions"][key]
        elif position.stop > s["current_stop"]:
            # Trail raised. A plain "stop" reason (price crossed the resting
            # stop) needs no action here -- the broker's own stop order
            # already covers it via main()'s normal reconciliation loop.
            print(f"  {ORLAND_SYMBOL}/orland_s1: raising stop {s['current_stop']:.2f} -> {position.stop:.2f}")
            if not args.dry_run:
                broker.cancel_order(s["stop_order_id"])
                s["stop_order_id"] = broker.submit_stop_sell(ORLAND_SYMBOL, s["shares"], position.stop)
            s["current_stop"] = position.stop
            s["trail_active"] = position.trail_active
            s["extreme"] = position.extreme_price

    # Entry check -- dedupe on the latest closed 5m bar, matching OrlandMagics'
    # own live_runner.py cursor convention (finer-grained than the daily-only
    # cursors breakout/mtf use, since this strategy's entries are timed off 5m).
    pattern_df = timeframe_data.get(strategy_s1.PATTERN_TIMEFRAME)
    cursor_tf = strategy_s1.PATTERN_TIMEFRAME if pattern_df is not None and not pattern_df.empty else "1D"
    cursor_df = pattern_df if cursor_tf == strategy_s1.PATTERN_TIMEFRAME else daily_df
    if cursor_df is None or cursor_df.empty:
        return
    latest_ts = cursor_df.index[-1]
    cursor_key = f"last_{cursor_tf}_orland_s1"
    if str(cursor.get(cursor_key)) == str(latest_ts):
        return  # already processed this bar
    cursor[cursor_key] = str(latest_ts)

    if key not in state["positions"]:
        signal = strategy_s1.check_entry(ORLAND_SYMBOL, timeframe_data, levels_by_tf)
        if signal is not None and signal.direction == "up":
            print(f"  {ORLAND_SYMBOL}: orland_s1 entry signal" + (" (entries halted)" if halt_entries else ""))
            if not (args.dry_run or halt_entries):
                atr_val = float(atr(daily_df["High"], daily_df["Low"], daily_df["Close"]).iloc[-1])
                entry = place_entry("orland_s1", ORLAND_SYMBOL, signal.entry_price, signal.stop,
                                     atr_val, equity, ORLAND_RISK_PCT, extra={"direction": signal.direction})
                if entry:
                    state["positions"][key] = entry


def run_orland_elder(state, equity, halt_entries, args):
    """System 2 (Elder's Triple Screen), ported from OrlandMagics. Long-only,
    same reasoning as run_orland_s1. Daily cadence -- cursor dedupes on the
    latest daily bar, matching run_breakout's pattern. Fixed protective stop
    from entry (the chosen exit rule, adx_fade, never trails), so unlike
    run_orland_s1 there's no stop-raise branch to handle here."""
    print("\n--- Orland Elder Triple Screen (EWZ) ---")
    daily_map = broker.fetch_daily_bars([ORLAND_SYMBOL], lookback_days=800)
    daily_df = daily_map.get(ORLAND_SYMBOL)
    if daily_df is None or daily_df.empty:
        return
    weekly_df = orland_data.drop_unclosed_bar(orland_data.resample_ohlc(daily_df, "W"), "1W")

    key = f"{ORLAND_SYMBOL}|orland_elder"
    cursor = state["cursors"].setdefault(ORLAND_SYMBOL, {})

    if key in state["positions"]:
        s = state["positions"][key]
        position = elder_strategy.ElderPosition(
            direction=s.get("direction", "up"), entry_price=s["entry_price"],
            stop=s["current_stop"], initial_stop=s["initial_stop"], entry_atr=s["atr_at_entry"],
        )
        position.trail_active = s.get("trail_active", False)
        position.extreme_price = s.get("extreme", s["entry_price"])

        atr_val = float(atr(daily_df["High"], daily_df["Low"], daily_df["Close"]).iloc[-1])
        bar = daily_df.iloc[-1]
        context = {"atr_val": atr_val, "df": daily_df}
        exit_rule = elder_exits.make_adx_fade(threshold=25)  # the cross-asset-generalizing pick, see FINDINGS
        reason = exit_rule(position, bar, context)

        if reason == "adx_fade":
            print(f"  {ORLAND_SYMBOL}/orland_elder: adx_fade -- closing")
            if not args.dry_run:
                broker.cancel_order(s["stop_order_id"])
                fill = broker.submit_market_sell(ORLAND_SYMBOL, s["shares"])
                exit_price = fill["fill_price"] if fill.get("filled") else float(bar["Close"])
                record_closed_trade(state, s, exit_price, log_key="orland_trade_log")
            del state["positions"][key]
        # reason == "stop" needs no action here -- the broker's resting stop
        # order + main()'s normal reconciliation loop already cover it.

    today_str = str(daily_df.index[-1].date())
    if cursor.get("last_daily_orland_elder") == today_str:
        return
    cursor["last_daily_orland_elder"] = today_str

    if key not in state["positions"]:
        signal = elder_strategy.check_entry(daily_df, weekly_df)
        if signal is not None and signal.direction == "up":
            print(f"  {ORLAND_SYMBOL}: orland_elder entry signal" + (" (entries halted)" if halt_entries else ""))
            if not (args.dry_run or halt_entries):
                atr_val = float(atr(daily_df["High"], daily_df["Low"], daily_df["Close"]).iloc[-1])
                entry = place_entry("orland_elder", ORLAND_SYMBOL, signal.entry_price, signal.stop,
                                     atr_val, equity, ORLAND_RISK_PCT, extra={"direction": signal.direction})
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
    state.setdefault("trade_log", [])
    state.setdefault("orland_trade_log", [])

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
            log_key = "orland_trade_log" if key.endswith(("|orland_s1", "|orland_elder")) else "trade_log"
            record_closed_trade(state, s, status["fill_price"], log_key=log_key)
            del state["positions"][key]
        elif status["status"] in ("canceled", "expired", "rejected"):
            print(f"  {key}: stop order {status['status']} unexpectedly -- clearing local state, check the account manually")
            del state["positions"][key]

    # Peak-equity circuit breaker + daily loss gate: both only ever block NEW
    # entries below (run_* still manage/trail/take-profit existing positions
    # regardless -- halting means "stop adding risk", not "abandon positions").
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

    orland_halt = halt_entries or orland_entries_halted(state)
    if orland_halt and not halt_entries:
        print(f"\n!!! ORLAND HALT: {ORLAND_CONSEC_LOSS_HALT} consecutive EWZ losses -- "
              f"new orland entries halted (breakout/mtf/pcse unaffected)")

    # Each run_* call is independent (different strategy, often different
    # symbols) -- one raising shouldn't take the rest down with it. Before
    # this, an uncaught exception from ANY strategy (e.g. a data-fetch error)
    # aborted main() entirely, skipping save_state below -- so whatever
    # earlier strategies in this same pass had already done (fills, trail
    # raises, closes) never got persisted to local state, only to resurface
    # next run as a phantom re-signal that collides with the still-open
    # broker-side order ("wash trade detected") and repeats the failure
    # forever. Order-submission errors specifically no longer raise at all
    # (see broker.submit_market_buy) -- this is defense in depth for anything
    # else (bad data, a network blip) that still could.
    try:
        run_breakout(state, equity, risk_pct, halt_entries, args)
        run_swing_breakout(state, equity, risk_pct, halt_entries, args)
        any_open_mtf = any(k.endswith("|mtf") for k in state["positions"])
        if MTF_ENTRIES_DISABLED and not any_open_mtf:
            # Skip the fetch entirely: no position to manage and no new entries
            # will ever be placed, so there's nothing for this call to do --
            # avoids ~minutes of Alpaca API load every run (400-day, 40-symbol
            # 4H-bar fetch, slow under IEX free-tier pagination) for zero benefit.
            print("\n--- MTF v2: entries disabled, no open position -- skipped ---")
        else:
            run_mtf(state, equity, risk_pct, halt_entries or MTF_ENTRIES_DISABLED, args)
        run_pcse(state, equity, risk_pct, halt_entries, args)
        run_orland_s1(state, equity, orland_halt, args)
        run_orland_elder(state, equity, orland_halt, args)
    except Exception as e:
        print(f"\n!!! run aborted by unexpected error: {e!r} -- saving state for whatever completed above")
    finally:
        if not args.dry_run:
            save_state(state)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
