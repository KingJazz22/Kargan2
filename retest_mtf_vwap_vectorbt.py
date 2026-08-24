"""vectorbt-data-sourced rerun of retest_mtf_vwap_full.py.

Context: retest_mtf_vwap_full.py (deep 10.5yr Alpaca history) hung for the
prior agent -- diagnosed here as NOT an infinite hang but Alpaca's IEX free
feed paginating 4H bars in ~2-3 week chunks regardless of the `limit=10000`
request param (confirmed via urllib3 DEBUG logging: consecutive page_tokens
for SPY's 4H series are ~16-19 days apart), forcing ~200+ sequential paginated
HTTP round-trips per symbol for the 4H leg alone (~60s/symbol measured
directly). Across 40 symbols x (daily + 4H) that is ~40+ minutes wall clock,
almost all of it network I/O wait -- consistent with the "13 min wall clock,
18s CPU" signature the killed run left behind. It was never stuck; it just
needed longer than the process was given.

Per explicit instruction, this script instead sources bars via vectorbt's own
data layer (vbt.YFData, yfinance-backed) to both sidestep that slow path and
fulfill the ask to use vectorbt for the data step. Trade-off (unavoidable,
inherent to yfinance, not a vectorbt limitation): Yahoo hard-caps 60m/1H bars
at the last ~730 calendar days (confirmed empirically -- a 729d request
succeeded, a 730d+/800d request raised "must be within the last 730 days"),
so the 4H leg here only covers ~2.9 years, not the original's 10.5 years. The
daily leg (used for both the Breakout Hunter strand and the VWAP(200) trend
filter) has no such cap and goes back to each symbol's IPO.

Everything downstream of the fetch is UNCHANGED project code: strategy_mtf.py
for signal generation, simulate_live_system.run_combined for the shared
40-symbol/$9,995.18-capital-pool execution engine (the exact engine that
produced the pre-fix live-relevant number this is meant to update: t=+0.98,
n=121, FINDINGS_MTF.md's "Deep-history VWAP(200) full-sample retest"), and
sweep_mtf_trend_filter.mtf_stats for the pooled significance stats.
"""
import sys
import time
import types

import vectorbt as vbt

# simulate_live_system.py unconditionally imports alpaca_fetch at module level
# (`from alpaca_fetch import fetch_stock_daily, fetch_stock_4h`), and
# alpaca_fetch.py in turn imports python-dotenv + alpaca-py -- neither of
# which is installed in the .venv this vectorbt retest runs under (that venv
# only has the vectorbt-pilot deps). We never actually call the Alpaca
# fetchers here (raw_cache is pre-populated below, and fetch_raw() skips any
# symbol already in raw_cache), so stub the module out rather than pulling
# Alpaca's dependency chain into the vectorbt venv just to satisfy an unused
# import.
_alpaca_stub = types.ModuleType("alpaca_fetch")


def _unavailable(*_a, **_k):
    raise RuntimeError("alpaca_fetch stubbed out for the vectorbt-sourced retest -- "
                        "raw_cache is pre-populated via vectorbt/yfinance, this should never be called")


_alpaca_stub.fetch_stock_daily = _unavailable
_alpaca_stub.fetch_stock_4h = _unavailable
sys.modules.setdefault("alpaca_fetch", _alpaca_stub)

import simulate_live_system as sim
from basket import US_SYMBOLS
from sweep_mtf_trend_filter import mtf_stats
from test_mtf_oos_symbols import OOS_US_SYMBOLS

SYMBOLS = US_SYMBOLS + OOS_US_SYMBOLS

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_daily_vbt(symbol: str):
    df = vbt.YFData.download(symbol, period="max", interval="1d").get()
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[OHLCV_COLS].dropna()


def fetch_4h_vbt(symbol: str, period_days: int = 729):
    """Same resample-from-1H convention as data.py's fetch_yfinance_4h, just
    sourced through vbt.YFData instead of a bare yf.download call."""
    df = vbt.YFData.download(symbol, period=f"{period_days}d", interval="60m").get()
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[OHLCV_COLS].dropna()
    resampled = df.resample("4h", origin="epoch").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    })
    return resampled.dropna()


def fetch_raw_vectorbt(symbols):
    """Same shape/contract as simulate_live_system.fetch_raw: {symbol: (daily_df, ltf_4h_df)},
    but sourced via vectorbt's YFData instead of Alpaca."""
    raw_cache = {}
    for symbol in symbols:
        daily_raw, ltf_raw = None, None
        try:
            daily_raw = fetch_daily_vbt(symbol)
        except Exception as e:
            print(f"  skip fetch daily/{symbol}: {e}")
        try:
            ltf_raw = fetch_4h_vbt(symbol)
        except Exception as e:
            print(f"  skip fetch 4h/{symbol}: {e}")
        raw_cache[symbol] = (daily_raw, ltf_raw)
        d_rows = 0 if daily_raw is None else len(daily_raw)
        l_rows = 0 if ltf_raw is None else len(ltf_raw)
        span = f"{ltf_raw.index.min().date()} -> {ltf_raw.index.max().date()}" if l_rows else "n/a"
        print(f"  fetched {symbol:6s} daily_rows={d_rows:5d} 4h_rows={l_rows:5d} 4h_span={span}")
    return raw_cache


def fmt_row(label, s):
    return (f"{label:<24}{s['n']:>6}{s['win_rate']*100:>8.1f}%{s['avg_r']:>+9.3f}"
            f"{s['t_stat']:>+8.2f}{s['pnl']:>+12,.2f}")


def main():
    t0 = time.time()
    print(f"Fetching daily+4H bars via vectorbt YFData for all {len(SYMBOLS)} symbols "
          f"(daily: full history via yfinance; 4H: resampled from 60m, ~last 730d yfinance cap)...")
    raw_cache = fetch_raw_vectorbt(SYMBOLS)
    print(f"\nFetch done in {time.time()-t0:.1f}s")

    header = f"{'variant':<24}{'n':>6}{'win%':>9}{'avg R':>9}{'t-stat':>8}{'PnL':>12}"
    print(f"\n{'='*70}\nvectorbt-data-sourced 40-SYMBOL COMBINED PORTFOLIO -- MTF strand only\n{'='*70}")
    print(header)
    print("-" * len(header))

    print("\nRunning baseline (no trend filter)...")
    trades_base, eq_base = sim.run_combined(SYMBOLS, sim.STARTING_EQUITY, sim.RISK_PCT,
                                             sim.MAX_CONCURRENT_POSITIONS, verbose=False,
                                             mtf_trend_filter=None, raw_cache=raw_cache)
    s_base = mtf_stats(trades_base)
    print(fmt_row("baseline (unfiltered)", s_base))

    print("\nRunning VWAP(200)-filtered (MTF v2 live config)...")
    trades_vwap, eq_vwap = sim.run_combined(SYMBOLS, sim.STARTING_EQUITY, sim.RISK_PCT,
                                             sim.MAX_CONCURRENT_POSITIONS, verbose=False,
                                             mtf_trend_filter=("vwap", 200), raw_cache=raw_cache)
    s_vwap = mtf_stats(trades_vwap)
    print(fmt_row("vwap_200", s_vwap))

    print(f"\n{'='*70}\nFull portfolio equity curves (Breakout Hunter + MTF combined)\n{'='*70}")
    for label, eq in (("baseline", eq_base), ("vwap_200", eq_vwap)):
        if eq.empty:
            print(f"{label}: no data")
            continue
        final = eq["equity"].iloc[-1]
        total_ret = (final / sim.STARTING_EQUITY - 1) * 100
        running_max = eq["equity"].cummax()
        dd = ((eq["equity"] - running_max) / running_max).min() * 100
        print(f"{label:<12} final_equity=${final:,.2f}  total_return={total_ret:+.1f}%  max_dd={dd:.1f}%")

    print(f"\n{'='*70}\nVERDICT (vectorbt-data-sourced, ~2.9yr yfinance window vs original's 10.5yr Alpaca window)\n{'='*70}")
    print(f"baseline  : n={s_base['n']:4d}  avg_R={s_base['avg_r']:+.3f}  t={s_base['t_stat']:+.2f}")
    print(f"vwap_200  : n={s_vwap['n']:4d}  avg_R={s_vwap['avg_r']:+.3f}  t={s_vwap['t_stat']:+.2f}")
    if s_vwap['n'] > 0:
        improved = s_vwap['avg_r'] > s_base['avg_r']
        significant = abs(s_vwap['t_stat']) > 2.0
        print(f"\nVWAP(200) {'improves' if improved else 'does NOT improve'} avg_R vs baseline "
              f"on this vectorbt-sourced sample.")
        print(f"VWAP(200) result is {'STATISTICALLY SIGNIFICANT' if significant else 'NOT statistically significant'} "
              f"on its own (|t|={abs(s_vwap['t_stat']):.2f}).")
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
