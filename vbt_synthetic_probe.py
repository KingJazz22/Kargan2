"""Standalone probe against vectorbt 1.1.0's actual installed behavior, using a
handful of synthetic OHLC bars with entries/exits/stops I construct by hand.
Goal: empirically confirm (1) SizeType.Percent sizing formula, (2) the 1-bar
signal-shift convention, (3) adjust_sl_func_nb's calling convention and how to
replicate the original engine's running-max-High extreme + frozen-ATR-at-entry
+ live-ATR-while-trailing ratchet using a fully self-managed absolute stop
(sl_trail always False, fraction recomputed every bar relative to the frozen
entry reference price).
"""
import numpy as np
import pandas as pd
from numba import njit

import vectorbt as vbt
from vectorbt.portfolio.enums import AdjustSLContext, SizeType, StopEntryPrice

print("=" * 70)
print("PROBE 1: SizeType.Percent sizing formula")
print("=" * 70)

# 10 flat-ish bars, single entry at bar 2 (shifted from a signal at bar 1).
n = 10
idx = pd.date_range("2024-01-01", periods=n, freq="D")
close = pd.Series([100.0] * n, index=idx)
open_ = pd.Series([100.0] * n, index=idx)
high = pd.Series([101.0] * n, index=idx)
low = pd.Series([99.0] * n, index=idx)

entries = pd.Series([False] * n, index=idx)
entries.iloc[2] = True  # fires at bar 2's Open = 100.0
exits = pd.Series([False] * n, index=idx)
exits.iloc[8] = True

RISK_PCT = 0.01
stop_distance = 5.0  # entry_price(100) - initial_stop(95)
size_frac = RISK_PCT * 100.0 / stop_distance  # = 0.01*100/5 = 0.2
size_arr = np.full(n, np.nan)
size_arr[2] = size_frac

pf = vbt.Portfolio.from_signals(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    price=open_,
    size=size_arr, size_type=SizeType.Percent,
    init_cash=100_000.0, freq="1D",
)
recs = pf.trades.records_readable
print(recs[["Entry Timestamp", "Avg Entry Price", "Size", "Exit Timestamp", "Avg Exit Price"]])
predicted_shares = RISK_PCT * 100_000.0 / stop_distance  # cash=100k * 1% / stop_distance
print(f"predicted shares (risk_amount/stop_distance, cash=100k) = {predicted_shares}")
print(f"actual shares from vbt = {recs['Size'].iloc[0]}")
print(f"match = {abs(recs['Size'].iloc[0] - predicted_shares) < 1e-6}")

print()
print("=" * 70)
print("PROBE 2: entry fires at price[i] for entries[i]=True -- confirm shift convention")
print("=" * 70)
# entries.iloc[2]=True with price=open_ -> does it fill at bar 2's Open (100) or bar 3's?
print(recs[["Entry Timestamp"]])
print("Entry Timestamp should be index[2] =", idx[2], "if same-bar/no-implicit-shift")

print()
print("=" * 70)
print("PROBE 3: adjust_sl_func_nb calling convention -- does it get called on the entry bar?")
print("=" * 70)

call_log = []


@njit
def logging_adjust_sl_func_nb(c, log_i, log_col, log_ii, log_posnow, log_currstop, log_currtrail, log_currprice, log_ptr):
    idx_ = log_ptr[0]
    if idx_ < log_i.shape[0]:
        log_i[idx_] = c.i
        log_col[idx_] = c.col
        log_ii[idx_] = c.init_i
        log_posnow[idx_] = c.position_now
        log_currstop[idx_] = c.curr_stop
        log_currtrail[idx_] = c.curr_trail
        log_currprice[idx_] = c.curr_price
        log_ptr[0] = idx_ + 1
    return c.curr_stop, c.curr_trail


MAXLOG = 200
log_i = np.full(MAXLOG, -999, dtype=np.int64)
log_col = np.full(MAXLOG, -999, dtype=np.int64)
log_ii = np.full(MAXLOG, -999, dtype=np.int64)
log_posnow = np.full(MAXLOG, np.nan)
log_currstop = np.full(MAXLOG, np.nan)
log_currtrail = np.zeros(MAXLOG, dtype=np.bool_)
log_currprice = np.full(MAXLOG, np.nan)
log_ptr = np.zeros(1, dtype=np.int64)

pf2 = vbt.Portfolio.from_signals(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    price=open_,
    size=size_arr, size_type=SizeType.Percent,
    sl_stop=0.05, sl_trail=False,
    stop_entry_price=StopEntryPrice.FillPrice,
    adjust_sl_func_nb=logging_adjust_sl_func_nb,
    adjust_sl_args=(log_i, log_col, log_ii, log_posnow, log_currstop, log_currtrail, log_currprice, log_ptr),
    init_cash=100_000.0, freq="1D",
)
ncalls = log_ptr[0]
print(f"adjust_sl_func_nb called {ncalls} times for {n} bars (single column)")
for k in range(ncalls):
    print(f"  bar i={log_i[k]}  init_i={log_ii[k]}  position_now={log_posnow[k]}  "
          f"curr_stop={log_currstop[k]:.4f}  curr_trail={log_currtrail[k]}  curr_price={log_currprice[k]}")
print("Entry bar was index 2. Check: is adjust_sl called for i=2 with position_now>0 (same-bar) or position_now==0 (pre-entry)?")
recs2 = pf2.trades.records_readable
print(recs2[["Entry Timestamp", "Avg Entry Price", "Exit Timestamp", "Avg Exit Price", "Status"]])

print()
print("=" * 70)
print("PROBE 4: does curr_price ever update when sl_trail is always False?")
print("=" * 70)
# bars after entry (i=3..) should all show curr_price == entry fill price (100.0) if never updated.
print("curr_price values seen at i>=3:", [log_currprice[k] for k in range(ncalls) if log_i[k] >= 3])
