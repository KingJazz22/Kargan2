"""Vectorbt 1.1.0 port of backtest_swing_breakout.py's event-driven engine.

Reuses strategy_swing_breakout.prepare(df) UNCHANGED for entries/atr/support,
so entry signals are guaranteed identical to the original -- this isolates the
comparison to the execution engine only.

Trailing-stop mechanics (see report / FINDINGS for the full derivation, worth
summarizing here since it's non-obvious from vectorbt's docs alone):

  - vectorbt's adjust_sl_func_nb returns a FRACTION relative to a reference
    price (`curr_price`) that vectorbt itself tracks per-column. That
    reference can be frozen at the entry fill price (sl_trail=False) or
    auto-ratcheted to the running max High (sl_trail=True) -- but the
    ratchet-with-High update always happens strictly AFTER the callback
    returns for that bar, i.e. `curr_price` passed into the callback for bar
    i has NOT yet folded in bar i's own High. It also NEVER folds in the
    entry bar's own High at all, since adjust_sl_func_nb is not invoked with
    an open position on the entry bar itself (confirmed empirically: entries
    are only visible to the callback starting the bar AFTER entry).
  - A frozen (sl_trail=False) reference is mathematically incompatible with
    this strategy: `get_stop_price_nb` hardcodes hit_below=True for the SL
    leg of a long position, which requires the returned fraction to be >= 0
    (stop <= reference). Once the trailing stop ratchets above the entry
    price -- which happens routinely in any profitable trending trade --
    the frozen-reference fraction goes negative and vectorbt raises
    ValueError("Stop value must be 0 or greater"). Discovered by hitting this
    error directly on synthetic data (vbt_synthetic_probe2.py, Case B).
  - Fix: use sl_trail=True (let vectorbt auto-ratchet curr_price using raw
    High) so the reference always grows, AND inside the callback compute the
    candidate trail level using max(curr_price, high[today]) -- manually
    folding in today's own High before it's "officially" absorbed by
    vectorbt's own bookkeeping -- then clamp the final fraction to >= 0.
    This exactly reproduces the original's same-bar "extreme = max(extreme,
    High[today])" update in the cases tested, and only degrades gracefully
    (clamped to a stop exactly at curr_price) in the rare case a single bar's
    target jump would overshoot even that. Verified byte-for-byte identical
    to a hand-transcribed copy of the original's bar loop on 4 synthetic
    cases (initial-stop exit, ratchet exit, late-activation-with-varying-ATR
    exit, gap-through-clamped-to-open) in vbt_synthetic_probe2.py.
  - One known, accepted residual gap (vbt_synthetic_probe2.py Case E): if the
    ENTRY bar's own High is the single highest point the trade ever reaches
    (an immediate post-breakout spike-and-reverse), the original's trail
    level reflects that entry-day spike (since original updates `extreme`
    within the very same loop iteration as the entry) but the port's does
    not (since the callback is never invoked with an open position on the
    entry bar, so there is no hook to fold that day's High in at all). This
    can only matter for trades that would have exited on the FIRST bar after
    entry due to a trail set by the entry bar's own spike; expected to be
    rare and is called out explicitly in the validation report.
  - Also NOT modeled here (per the porting plan, an accepted divergence):
    the original's independent `emergency_stop` (initial_stop - 2*ATR, a
    hard secondary stop checked first each bar) and same-bar entry+exit
    (the original can open AND stop out a position within the same calendar
    bar since entry-fill and stop-management run in the same loop iteration;
    vectorbt's from_signals pipeline structurally cannot check a stop on the
    same bar a position is opened by a signal, since adjust_sl_func_nb runs
    BEFORE that bar's order is processed).
"""
import pickle
import time

import numpy as np
import pandas as pd
from numba import njit

import strategy_swing_breakout as strat
import vectorbt as vbt
from basket import US_SYMBOLS
from fetch_cache_swing_breakout import load_cached
from test_mtf_oos_symbols import OOS_US_SYMBOLS
from vectorbt.portfolio.enums import SizeType, StopEntryPrice

WARMUP_BARS = 2 * strat.SWING_LEFT + 2 * strat.SWING_RIGHT + 50
RISK_PCT = 0.01
STARTING_EQUITY = 100_000.0

TRAIL_ACTIVATE_ATR = strat.TRAIL_ACTIVATE_ATR
TRAIL_ATR_MULT = strat.TRAIL_ATR_MULT
INITIAL_STOP_ATR_BUFFER = strat.INITIAL_STOP_ATR_BUFFER


@njit(cache=True)
def adjust_sl_func_nb(c, high_, close_, atr_live_, initial_stop_arr_, atr_at_entry_arr_,
                       trail_active_, trail_stop_, last_init_i_,
                       trail_activate_atr, trail_atr_mult):
    if c.position_now == 0:
        last_init_i_[c.col] = -1
        return c.curr_stop, c.curr_trail

    if last_init_i_[c.col] != c.init_i:
        trail_active_[c.col] = False
        trail_stop_[c.col] = np.nan
        last_init_i_[c.col] = c.init_i

    atr_at_entry = atr_at_entry_arr_[c.init_i]
    favorable = close_[c.i] - c.init_price
    if not trail_active_[c.col] and favorable >= trail_activate_atr * atr_at_entry:
        trail_active_[c.col] = True
    if trail_active_[c.col]:
        extreme_today = max(c.curr_price, high_[c.i])
        candidate = extreme_today - trail_atr_mult * atr_live_[c.i]
        trail_stop_[c.col] = candidate if np.isnan(trail_stop_[c.col]) else max(trail_stop_[c.col], candidate)

    active_stop = trail_stop_[c.col] if trail_active_[c.col] else initial_stop_arr_[c.init_i]
    frac = max(1.0 - active_stop / c.curr_price, 0.0)
    return frac, True


def build_port_arrays(df: pd.DataFrame):
    """Reuses strat.prepare(df) unchanged, then builds the shifted (signal ->
    next-bar-fill) arrays the vectorbt engine needs."""
    data = strat.prepare(df)
    n = len(data)

    long_entry = data["long_entry"].to_numpy(dtype=bool).copy()
    long_entry[:WARMUP_BARS] = False  # match original's rows = data.iloc[WARMUP_BARS:]

    support = data["support"].to_numpy(dtype=np.float64)
    atr_arr = data["atr"].to_numpy(dtype=np.float64)
    open_arr = data["Open"].to_numpy(dtype=np.float64)
    high_arr = data["High"].to_numpy(dtype=np.float64)
    close_arr = data["Close"].to_numpy(dtype=np.float64)

    # Signal on close at bar T (long_entry[T]) fires at bar T+1's Open.
    entries = np.zeros(n, dtype=bool)
    entries[1:] = long_entry[:-1]

    initial_stop_arr = np.full(n, np.nan)
    atr_at_entry_arr = np.full(n, np.nan)
    sl_stop_arr = np.full(n, np.nan)
    size_arr = np.full(n, np.nan)

    for i in range(1, n):
        if not entries[i]:
            continue
        signal_support = support[i - 1]
        signal_atr = atr_arr[i - 1]
        entry_price = open_arr[i]
        if np.isnan(signal_support) or np.isnan(signal_atr) or np.isnan(entry_price):
            entries[i] = False
            continue
        init_stop = signal_support - INITIAL_STOP_ATR_BUFFER * signal_atr
        stop_distance = entry_price - init_stop
        if stop_distance <= 0:
            entries[i] = False
            continue
        size_frac = min(RISK_PCT * entry_price / stop_distance, 1.0)
        if size_frac <= 0:
            entries[i] = False
            continue
        initial_stop_arr[i] = init_stop
        atr_at_entry_arr[i] = signal_atr
        sl_stop_arr[i] = 1.0 - init_stop / entry_price
        size_arr[i] = size_frac

    return data, entries, initial_stop_arr, atr_at_entry_arr, sl_stop_arr, size_arr, high_arr, close_arr, atr_arr


def run_symbol(symbol: str, df: pd.DataFrame):
    (data, entries, initial_stop_arr, atr_at_entry_arr, sl_stop_arr, size_arr,
     high_arr, close_arr, atr_arr) = build_port_arrays(df)
    n = len(data)
    if not entries.any():
        return None

    exits = np.zeros(n, dtype=bool)  # long-only, exit is entirely stop-driven

    trail_active_state = np.zeros(1, dtype=np.bool_)
    trail_stop_state = np.full(1, np.nan)
    last_init_i_state = np.full(1, -1, dtype=np.int64)

    pf = vbt.Portfolio.from_signals(
        close=data["Close"], open=data["Open"], high=data["High"], low=data["Low"],
        entries=pd.Series(entries, index=data.index),
        exits=pd.Series(exits, index=data.index),
        price=data["Open"],
        size=size_arr, size_type=SizeType.Percent,
        sl_stop=sl_stop_arr, sl_trail=True,
        stop_entry_price=StopEntryPrice.FillPrice,
        adjust_sl_func_nb=adjust_sl_func_nb,
        adjust_sl_args=(high_arr, close_arr, atr_arr, initial_stop_arr, atr_at_entry_arr,
                         trail_active_state, trail_stop_state, last_init_i_state,
                         float(TRAIL_ACTIVATE_ATR), float(TRAIL_ATR_MULT)),
        init_cash=STARTING_EQUITY,
        freq="1D",
    )

    records = pf.trades.values  # raw structured numpy array (pf.trades.records is a DataFrame)
    if len(records) == 0:
        return None

    closed = records[records["status"] == 1]  # TradeStatus.Closed
    if len(closed) == 0:
        return None

    index = data.index.values
    rows = []
    for rec in closed:
        entry_idx = int(rec["entry_idx"])
        exit_idx = int(rec["exit_idx"])
        entry_price = float(rec["entry_price"])
        exit_price = float(rec["exit_price"])
        size = float(rec["size"])
        atr_at_entry = float(atr_at_entry_arr[entry_idx])
        init_stop = float(initial_stop_arr[entry_idx])
        stop_distance = entry_price - init_stop
        risk_amount = stop_distance * size

        # Re-derive exit_reason with the same sticky semantics as the
        # original: "trailing_stop" if Close ever reached entry_price +
        # 1.0*atr_at_entry between entry and exit (inclusive), else
        # "initial_stop".
        window_close = close_arr[entry_idx:exit_idx + 1]
        went_favorable = np.any(window_close - entry_price >= TRAIL_ACTIVATE_ATR * atr_at_entry)
        exit_reason = "trailing_stop" if went_favorable else "initial_stop"

        pnl = (exit_price - entry_price) * size
        r_multiple = pnl / risk_amount if risk_amount != 0 else 0.0

        rows.append({
            "symbol": symbol,
            "entry_date": pd.Timestamp(index[entry_idx]),
            "entry_price": entry_price,
            "shares": size,
            "initial_stop": init_stop,
            "atr_at_entry": atr_at_entry,
            "risk_amount": risk_amount,
            "exit_date": pd.Timestamp(index[exit_idx]),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "r_multiple": r_multiple,
        })

    return pd.DataFrame(rows)


def pooled_stats(trades_df: pd.DataFrame):
    """Same formulas as basket.run_basket -- ddof=1 for stdev/t-stat SE."""
    if trades_df is None or len(trades_df) == 0:
        return {"num_trades": 0}
    r_mults = trades_df["r_multiple"].to_numpy()
    pnls = trades_df["pnl"].to_numpy()
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = 100 * len(wins) / len(trades_df)
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    se = r_mults.std(ddof=1) / np.sqrt(len(r_mults)) if len(r_mults) > 1 else 0.0
    t_stat = r_mults.mean() / se if se > 0 else 0.0

    return {
        "num_trades": len(trades_df),
        "win_rate_pct": round(win_rate, 1),
        "avg_r_multiple": round(float(r_mults.mean()), 3),
        "median_r_multiple": round(float(np.median(r_mults)), 3),
        "stdev_r_multiple": round(float(r_mults.std()), 3),
        "profit_factor": round(profit_factor, 2),
        "t_stat_vs_zero": round(float(t_stat), 2),
    }


def run_group(group_name, symbols, out_path):
    print(f"\n=== Swing-structure breakout (VECTORBT port), {group_name} ===")
    all_trades = []
    for symbol in symbols:
        df = load_cached(symbol)
        try:
            trades_df = run_symbol(symbol, df)
        except Exception as e:
            print(f"  skip {symbol}: {type(e).__name__}: {e}")
            continue
        n_trades = 0 if trades_df is None else len(trades_df)
        print(f"  {group_name:6s} {symbol:9s} trades={n_trades:3d}")
        if trades_df is not None:
            all_trades.append(trades_df)

    if not all_trades:
        print("  no trades in this group")
        stats = {"num_trades": 0}
        pooled = pd.DataFrame()
    else:
        pooled = pd.concat(all_trades, ignore_index=True)
        stats = pooled_stats(pooled)
        print(f"\nTotal pooled trades: {len(pooled)}")
        print("\n=== Pooled trade statistics (all symbols, R-multiples) ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("\n  exit reason breakdown:")
        print(pooled["exit_reason"].value_counts().to_string())

    with open(out_path, "wb") as f:
        pickle.dump({"group_name": group_name, "stats": stats, "trades": pooled}, f)
    print(f"Saved trades + stats to {out_path}")
    return stats


def main():
    t0 = time.time()
    is_stats = run_group("US", US_SYMBOLS, "swing_breakout_vectorbt_IS.pkl")
    oos_stats = run_group("US_OOS", OOS_US_SYMBOLS, "swing_breakout_vectorbt_OOS.pkl")
    elapsed = time.time() - t0

    print("\n\n=== SUMMARY (vectorbt port) ===")
    print(f"IS : {is_stats}")
    print(f"OOS: {oos_stats}")
    print(f"\nWall-clock runtime (both groups, includes first-call numba JIT compile): {elapsed:.2f}s")


if __name__ == "__main__":
    main()
