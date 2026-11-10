"""Breakout Hunter + VXX regime filter.

FINDINGS_ALL_BOTS.md found Breakout Hunter's own entries underperform random
entries through the same exit engine (10/10 random draws beat real, in R and
in percent terms) -- despite the raw strategy clearing significance in
FINDINGS_BREAKOUT.md. One plausible fix: gate entries by the broader market's
volatility regime. VXX (iPath Series B S&P 500 VIX Short-Term Futures ETN)
is used as a simple, always-available proxy for "the market is calm/complacent"
vs. "the market is fearful/choppy" -- a squeeze-breakout is a trend-continuation
bet that should do better when the tape isn't whipsawing.

Three independently-testable variants, combined with AND where more than one
flag is set:
  calm:    VXX close is below its own 20-day SMA (below-average fear)
  falling: VXX close is below where it was 5 trading days ago (fear easing)
  not_spiking: VXX close is not at (or above) its own 5-day high (no fresh
               vol spike right at entry)

This module does NOT change strategy_breakout.py -- it wraps it. `prepare()`
takes the already-prepared breakout frame (output of strat_bo.prepare) plus a
VXX daily OHLC frame aligned to the same symbol, and adds a filtered
`long_entry_vxx` column alongside the original `long_entry`, so a direct A/B
comparison can run against the exact same breakout signals.
"""
import pandas as pd

VXX_SMA_PERIOD = 20
VXX_FALLING_LOOKBACK = 5
VXX_SPIKE_LOOKBACK = 5


def _align_vxx(dates: pd.DatetimeIndex, vxx: pd.DataFrame) -> pd.DataFrame:
    """Reindex VXX onto the strategy's date index. Both trade on the NYSE
    calendar so this is normally an exact join; ffill covers the rare case
    where VXX is missing a date the equity symbol has (or vice versa)."""
    v = vxx.copy()
    v["vxx_sma"] = v["Close"].rolling(VXX_SMA_PERIOD).mean()
    v["vxx_falling"] = v["Close"] < v["Close"].shift(VXX_FALLING_LOOKBACK)
    v["vxx_spike_high"] = v["Close"].rolling(VXX_SPIKE_LOOKBACK).max()
    return v.reindex(dates).ffill()


def add_filters(prepared: pd.DataFrame, vxx: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of `prepared` (output of strat_bo.prepare) with VXX
    filter flag columns and three gated long-entry variants added."""
    out = prepared.copy()
    v = _align_vxx(out.index, vxx)

    calm = v["Close"] < v["vxx_sma"]
    falling = v["vxx_falling"].astype(bool)
    not_spiking = v["Close"] < v["vxx_spike_high"]

    out["vxx_calm"] = calm.fillna(False)
    out["vxx_falling"] = falling.fillna(False)
    out["vxx_not_spiking"] = not_spiking.fillna(False)

    base = out["long_entry"]
    out["long_entry_calm"] = (base & out["vxx_calm"]).fillna(False)
    out["long_entry_falling"] = (base & out["vxx_falling"]).fillna(False)
    out["long_entry_not_spiking"] = (base & out["vxx_not_spiking"]).fillna(False)
    out["long_entry_calm_falling"] = (base & out["vxx_calm"] & out["vxx_falling"]).fillna(False)
    out["long_entry_all3"] = (base & out["vxx_calm"] & out["vxx_falling"] & out["vxx_not_spiking"]).fillna(False)
    return out
