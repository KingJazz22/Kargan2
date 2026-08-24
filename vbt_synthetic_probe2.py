"""Probe 5: full trailing-stop replication test. Build synthetic OHLC bars by
hand (bypassing strategy_swing_breakout.prepare/swing_structure entirely --
that part is reused unchanged and is not what's being validated here), run the
ORIGINAL bar-by-bar stop logic (transcribed directly from
backtest_swing_breakout.py's step-2 block) in plain Python as ground truth,
then run the vectorbt port using an adjust_sl_func_nb callback and confirm the
exit date/price/reason match exactly.
"""
import numpy as np
import pandas as pd
from numba import njit

import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, StopEntryPrice

TRAIL_ACTIVATE_ATR = 1.0
TRAIL_ATR_MULT = 2.5
RISK_PCT = 0.01


def original_stop_loop(dates, open_, high, low, close, atr, entry_i, entry_price, initial_stop, atr_at_entry):
    """Transcribed unchanged from backtest_swing_breakout.py step 2, applied
    starting at entry_i (inclusive -- entry and management happen same bar in
    the original)."""
    extreme = entry_price
    trail_active = False
    trail_stop = None
    for i in range(entry_i, len(dates)):
        extreme = max(extreme, high[i])
        favorable = close[i] - entry_price
        if not trail_active and favorable >= TRAIL_ACTIVATE_ATR * atr_at_entry:
            trail_active = True
        if trail_active:
            candidate = extreme - TRAIL_ATR_MULT * atr[i]
            trail_stop = candidate if trail_stop is None else max(trail_stop, candidate)
        active_stop = trail_stop if trail_active else initial_stop
        reason = "trailing_stop" if trail_active else "initial_stop"
        if low[i] <= active_stop:
            fill = active_stop if open_[i] > active_stop else open_[i]
            return dates[i], fill, reason
    return None, None, None


def build_case(bars):
    """bars: list of dicts with Open/High/Low/Close/atr."""
    n = len(bars)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame(bars, index=idx)
    return idx, df


def run_vbt_port(idx, df, entry_i, initial_stop, atr_at_entry):
    n = len(df)
    open_ = df["Open"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    close = df["Close"].values.astype(np.float64)
    atr_live = df["atr"].values.astype(np.float64)

    entries = np.zeros(n, dtype=np.bool_)
    entries[entry_i] = True
    entry_price = open_[entry_i]
    stop_distance = entry_price - initial_stop
    size_frac_val = min(RISK_PCT * entry_price / stop_distance, 1.0)
    size_arr = np.full(n, np.nan)
    size_arr[entry_i] = size_frac_val
    initial_stop_arr = np.full(n, np.nan)
    initial_stop_arr[entry_i] = initial_stop
    atr_at_entry_arr = np.full(n, np.nan)
    atr_at_entry_arr[entry_i] = atr_at_entry
    sl_stop_arr = np.full(n, np.nan)
    sl_stop_arr[entry_i] = 1 - initial_stop / entry_price  # only used at fill bar

    # state arrays (mutable across calls), sized for 1 column.
    # NOTE: curr_price/extreme tracking is delegated ENTIRELY to vectorbt's
    # own native trailing-stop ratchet (sl_trail=True always) rather than a
    # separately-maintained extreme array -- see report for why: expressing a
    # trail level that can exceed the frozen entry price as a fraction
    # relative to a FROZEN reference goes negative and vectorbt raises
    # ("Stop value must be 0 or greater"). Deferring to vectorbt's own
    # curr_price (which only folds in a bar's High *after* that bar's stop
    # check, i.e. one-bar-lagged relative to today) guarantees curr_price is
    # always >= our computed target (algebraic proof in the report), so the
    # fraction is always safely non-negative -- at the cost of a documented
    # one-bar lag vs the original's same-bar extreme update.
    trail_active_state = np.zeros(1, dtype=np.bool_)
    trail_stop_state = np.full(1, np.nan)
    last_init_i_state = np.full(1, -1, dtype=np.int64)

    @njit
    def adjust_sl_func_nb(c, high_, close_, atr_live_, initial_stop_arr_, atr_at_entry_arr_,
                           trail_active_, trail_stop_, last_init_i_):
        if c.position_now == 0:
            last_init_i_[c.col] = -1
            return c.curr_stop, c.curr_trail

        if last_init_i_[c.col] != c.init_i:
            # fresh trade -- reset sticky state
            trail_active_[c.col] = False
            trail_stop_[c.col] = np.nan
            last_init_i_[c.col] = c.init_i

        atr_at_entry_ = atr_at_entry_arr_[c.init_i]
        favorable = close_[c.i] - c.init_price
        if not trail_active_[c.col] and favorable >= TRAIL_ACTIVATE_ATR * atr_at_entry_:
            trail_active_[c.col] = True
        if trail_active_[c.col]:
            # Fold in TODAY's own high (matching the original's same-bar
            # extreme update) even though vectorbt's own curr_price tracker
            # is one bar behind -- the true target can now exceed curr_price,
            # so the resulting fraction is clamped to >=0 below (degrades to
            # "stop at curr_price" on the rare bar where a single day's
            # target jump outruns curr_price entirely).
            extreme_today = max(c.curr_price, high_[c.i])
            candidate = extreme_today - TRAIL_ATR_MULT * atr_live_[c.i]
            trail_stop_[c.col] = candidate if np.isnan(trail_stop_[c.col]) else max(trail_stop_[c.col], candidate)

        active_stop = trail_stop_[c.col] if trail_active_[c.col] else initial_stop_arr_[c.init_i]
        frac = max(1.0 - active_stop / c.curr_price, 0.0)
        return frac, True

    pf = vbt.Portfolio.from_signals(
        close=df["Close"], open=df["Open"], high=df["High"], low=df["Low"],
        entries=pd.Series(entries, index=idx), exits=pd.Series(np.zeros(n, dtype=bool), index=idx),
        price=df["Open"],
        size=size_arr, size_type=SizeType.Percent,
        sl_stop=sl_stop_arr, sl_trail=True,
        stop_entry_price=StopEntryPrice.FillPrice,
        adjust_sl_func_nb=adjust_sl_func_nb,
        adjust_sl_args=(high, close, atr_live, initial_stop_arr, atr_at_entry_arr,
                         trail_active_state, trail_stop_state, last_init_i_state),
        init_cash=100_000.0, freq="1D",
    )
    return pf


def compare_case(name, bars, entry_i, initial_stop, atr_at_entry):
    idx, df = build_case(bars)
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    atr = df["atr"].values
    entry_price = open_[entry_i]

    exp_date, exp_fill, exp_reason = original_stop_loop(idx, open_, high, low, close, atr, entry_i, entry_price, initial_stop, atr_at_entry)

    pf = run_vbt_port(idx, df, entry_i, initial_stop, atr_at_entry)
    recs = pf.trades.records_readable
    print(f"--- {name} ---")
    print(f"  original: exit_date={exp_date}, exit_price={exp_fill}, reason={exp_reason}")
    if len(recs) == 0:
        print("  vbt: NO TRADE RECORDED")
    else:
        r = recs.iloc[0]
        print(f"  vbt     : exit_date={r['Exit Timestamp']}, exit_price={r['Avg Exit Price']}, status={r['Status']}")
        match = (exp_date == r["Exit Timestamp"]) and abs(exp_fill - r["Avg Exit Price"]) < 1e-6
        print(f"  MATCH: {match}")
    print()


# --- Case A: initial-stop exit (price never gets favorable enough to trail) ---
bars_a = [
    dict(Open=100, High=101, Low=99, Close=100, atr=2),   # entry bar (i=0)
    dict(Open=100, High=100.5, Low=98, Close=99, atr=2),
    dict(Open=99, High=99.5, Low=94.5, Close=95, atr=2),  # Low breaches initial_stop=95
    dict(Open=95, High=96, Low=93, Close=94, atr=2),
]
compare_case("Case A: initial_stop exit", bars_a, entry_i=0, initial_stop=95.0, atr_at_entry=2.0)

# --- Case B: trailing stop activates and ratchets up, then exits ---
bars_b = [
    dict(Open=100, High=101, Low=99, Close=100, atr=2),    # entry bar i=0
    dict(Open=100, High=103, Low=99.5, Close=103, atr=2),  # i=1: favorable=3>=2 -> trail activates; extreme=103; trail_stop=103-5=98
    dict(Open=103, High=108, Low=102, Close=107, atr=2),   # i=2: extreme=108; trail_stop=108-5=103
    dict(Open=107, High=109, Low=104, Close=106, atr=2),   # i=3: extreme=109; trail_stop=max(103,109-5=104)=104
    dict(Open=106, High=106, Low=103, Close=104, atr=2),   # i=4: Low(103) <= trail_stop(104) -> exit at 104
]
compare_case("Case B: trailing_stop exit (ratchet)", bars_b, entry_i=0, initial_stop=95.0, atr_at_entry=2.0)

# --- Case C: trail activates on a LATER bar (not bar 1), live ATR changes ---
bars_c = [
    dict(Open=100, High=100.5, Low=99, Close=100, atr=2),  # entry i=0
    dict(Open=100, High=100.5, Low=99.5, Close=100.2, atr=2),  # i=1 not favorable yet
    dict(Open=100.2, High=103, Low=100, Close=102.5, atr=1.5),  # i=2: favorable=2.5>=2(frozen atr_at_entry) -> trail active; extreme=103; trail_stop=103-2.5*1.5=99.25
    dict(Open=102.5, High=105, Low=101, Close=104, atr=1.0),   # i=3: extreme=105; trail_stop=max(99.25,105-2.5=102.5)=102.5
    dict(Open=104, High=104.5, Low=102, Close=103, atr=1.0),   # i=4: Low(102)<=102.5 -> exit at 102.5
]
compare_case("Case C: late trail activation + varying live ATR", bars_c, entry_i=0, initial_stop=95.0, atr_at_entry=2.0)

# --- Case D: gap-through the initial stop at Open (fill should clamp to Open) ---
bars_d = [
    dict(Open=100, High=100.5, Low=99, Close=100, atr=2),
    dict(Open=90, High=91, Low=89, Close=90, atr=2),   # gaps below initial_stop(95) at open
]
compare_case("Case D: gap-through initial stop (fill clamps to Open)", bars_d, entry_i=0, initial_stop=95.0, atr_at_entry=2.0)

# --- Case E: entry bar's own High is the all-time-high of the trade (a spike
# that immediately reverses) -- known edge case: vectorbt's curr_price never
# folds in the entry bar's own High (adjust_sl_func_nb is never called with
# position_now>0 on the entry bar itself), so a trade whose tightest trail
# level would have been set by the entry-day spike exits later/differently
# in the port than in the original. ---
bars_e = [
    dict(Open=100, High=115, Low=99, Close=101, atr=2),   # entry bar, huge spike high
    dict(Open=101, High=104, Low=100, Close=103, atr=2),  # original: extreme=115, trail=110, Low=100<=110 -> exits here
]
compare_case("Case E: entry-bar spike-high omission (known edge case)", bars_e, entry_i=0, initial_stop=95.0, atr_at_entry=2.0)
