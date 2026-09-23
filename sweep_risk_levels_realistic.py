"""Robustness re-check of sweep_risk_levels.py's headline numbers (+289% over
~10yrs at 1% risk) against two specific realism gaps in that first version:

1. SURVIVORSHIP BIAS: the live symbol universe (basket.US_SYMBOLS +
   test_mtf_oos_symbols.OOS_US_SYMBOLS) is 40 hand-picked, currently-thriving
   mega/large-caps (AAPL, MSFT, NVDA, SPY, QQQ, COST, ADBE, ...) -- every
   single one is famous specifically BECAUSE it did well. None went
   bankrupt, got delisted, or spent a decade as a value trap. This re-run
   adds ~35 symbols that are still listed (Alpaca needs a live ticker) but
   are known laggards, cyclicals, or names that round-tripped hard at some
   point in 2016-2026 -- T, GE, F, WBA, C, M, airlines, cruise lines, and
   several 2010s-growth-then-crash survivors (SNAP, PYPL, UBER, COIN, ...).
   This does NOT fully solve survivorship bias (a name that was delisted
   outright, e.g. via bankruptcy, has no Alpaca history left to backtest --
   that data simply doesn't exist for free anywhere in this project), but it
   removes the "every symbol is a first-ballot winner" selection effect.

2. ZERO MODELED TRANSACTION COSTS on breakout/swing_breakout: unlike PCSE
   (0.05%/fill), those two engines charge nothing for slippage/spread on
   entry or exit. Applied here as a flat round-trip haircut on realized
   pnl_pct at trade-generation time (kept out of backtest_breakout.py/
   backtest_swing_breakout.py themselves since those are shared, validated
   files used elsewhere in the project -- this cost only applies to THIS
   robustness re-check).

Runs three variants back to back so the effect of each change is visible on
its own: (A) original 40-symbol universe, no costs [reproduces the prior
sweep's numbers], (B) same 40 symbols + realistic costs, (C) expanded ~75/
~55-symbol universe + realistic costs.
"""
import pickle
from pathlib import Path

import sweep_risk_levels as base
from basket import US_SYMBOLS
from test_mtf_oos_symbols import OOS_US_SYMBOLS
from alpaca_fetch import fetch_stock_daily, fetch_stock_4h

ORIGINAL_SYMBOLS = US_SYMBOLS + OOS_US_SYMBOLS         # the 40 hand-picked mega/large-caps
ORIGINAL_PCSE_SYMBOLS = US_SYMBOLS                       # 20 of those 40

# Still-listed (Alpaca needs a live ticker), but NOT hand-picked for fame --
# legacy/value laggards, cyclicals hit hard by COVID, and 2010s growth names
# that crashed 60-90% and only partially (if at all) recovered. The point is
# to include names an investor in 2016 could NOT have known would do well.
LAGGARD_AND_TURNAROUND_SYMBOLS = [
    "T", "VZ", "GE", "F", "GM", "KHC", "WBA", "C", "M", "GPS", "KSS",
    "IBM", "MMM", "DOW", "HPQ", "DELL", "UPS", "FDX", "TGT", "LOW",
    "SBUX", "YUM", "CL", "KMB", "GIS", "K",
    "DAL", "AAL", "UAL", "CCL", "RCL",
    "SNAP", "PYPL", "PINS", "ETSY", "ROKU", "TWLO", "UBER", "LYFT", "ABNB", "COIN",
]

EXPANDED_SYMBOLS = ORIGINAL_SYMBOLS + LAGGARD_AND_TURNAROUND_SYMBOLS
EXPANDED_PCSE_SYMBOLS = ORIGINAL_PCSE_SYMBOLS + LAGGARD_AND_TURNAROUND_SYMBOLS

ROUND_TRIP_COST_PCT = 0.001   # 0.10% total (entry+exit) -- comparable to PCSE's 0.05%/fill

CACHE_FILE = Path("_sweep_risk_levels_bars_cache.pkl")   # reuse the original sweep's cache -- additive, keyed by (type, symbol)


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def build_trades(symbols, pcse_symbols, cost_pct, label):
    cache = load_cache()
    all_trades = []

    print(f"--- [{label}] Breakout Hunter + Swing-Structure Breakout (daily), {len(symbols)} symbols ---")
    for symbol in symbols:
        key = ("daily", symbol)
        if key not in cache:
            try:
                cache[key] = fetch_stock_daily(symbol)
            except Exception as e:
                print(f"  skip {symbol} daily: {e}")
                cache[key] = None
        df = cache[key]
        if df is None:
            continue

        if len(df) > base.bt_bo.WARMUP_BARS + 5:
            trades, _ = base.bt_bo.run_backtest(df, starting_equity=1e12, risk_pct=0.01)
            for t in trades:
                if t.exit_price is None:
                    continue
                all_trades.append({
                    "strategy_key": "breakout", "symbol": symbol,
                    "entry_time": base.to_utc(t.entry_date), "exit_time": base.to_utc(t.exit_date),
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "pnl_pct": (t.exit_price / t.entry_price - 1) - cost_pct,
                    "stop_distance": t.entry_price - t.initial_stop,
                })
        if len(df) > base.bt_swing.WARMUP_BARS + 5:
            trades, _ = base.bt_swing.run_backtest(df, starting_equity=1e12, risk_pct=0.01)
            for t in trades:
                if t.exit_price is None:
                    continue
                all_trades.append({
                    "strategy_key": "swing_breakout", "symbol": symbol,
                    "entry_time": base.to_utc(t.entry_date), "exit_time": base.to_utc(t.exit_date),
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "pnl_pct": (t.exit_price / t.entry_price - 1) - cost_pct,
                    "stop_distance": t.entry_price - t.initial_stop,
                })
    n_bo = sum(1 for t in all_trades if t["strategy_key"] == "breakout")
    n_sw = sum(1 for t in all_trades if t["strategy_key"] == "swing_breakout")
    print(f"    {n_bo} breakout trades, {n_sw} swing_breakout trades")

    print(f"--- [{label}] PCSE (4H), {len(pcse_symbols)} symbols ---")
    n_pcse = 0
    for symbol in pcse_symbols:
        key = ("4h", symbol)
        if key not in cache:
            try:
                cache[key] = fetch_stock_4h(symbol)
            except Exception as e:
                print(f"  skip {symbol} 4h: {e}")
                cache[key] = None
        df = cache[key]
        if df is None:
            continue
        if len(df) <= base.bt_pcse.WARMUP_BARS + 5:
            continue
        try:
            trades, _ = base.bt_pcse.run_backtest(df, starting_equity=1e12, risk_pct=0.01)
        except Exception as e:
            print(f"  skip {symbol} pcse: {e}")
            continue
        for t in trades:
            if t.exit_price is None:
                continue
            raw_pnl_pct = (t.exit_price / t.entry_price - 1) * (1 if t.side == "long" else -1)
            # PCSE's own engine already applies its own 0.05%/fill cost internally
            # (cost_per_trade_pct default in backtest_pcse_final.run_backtest) --
            # don't double-charge it here.
            all_trades.append({
                "strategy_key": "pcse", "symbol": symbol,
                "entry_time": base.to_utc(t.entry_date), "exit_time": base.to_utc(t.exit_date),
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "pnl_pct": raw_pnl_pct, "stop_distance": t.stop_distance,
            })
        n_pcse += sum(1 for t in trades if t.exit_price is not None)
    print(f"    {n_pcse} PCSE trades")

    save_cache(cache)
    all_trades.sort(key=lambda t: t["entry_time"])
    print(f"    total: {len(all_trades)} trades\n")
    return all_trades


def run_variant(label, symbols, pcse_symbols, cost_pct):
    print(f"\n{'#'*100}\n# VARIANT: {label}\n{'#'*100}")
    trades = build_trades(symbols, pcse_symbols, cost_pct, label)
    if not trades:
        print("No trades -- skipping.")
        return

    hdr = (f"{'risk%':>7}{'trades':>8}{'trades/yr':>11}{'win%':>7}{'return%':>10}"
           f"{'CAGR%':>9}{'maxDD%':>9}{'Sharpe':>8}{'blk_circ':>9}{'blk_day':>8}")
    print(hdr)
    print("-" * len(hdr))
    for rp in base.RISK_LEVELS:
        closed, eq_df, taken, b_circ, b_day = base.replay_shared_ledger(trades, rp)
        s = base.summarize_replay(rp, closed, eq_df, taken, b_circ, b_day)
        if s is None:
            continue
        print(f"{s['risk_pct']*100:>6.2f}%{s['n_trades']:>8}{s['trades_per_year']:>11.1f}"
              f"{s['win_rate']:>6.1f}%{s['total_return']:>9.1f}%{s['cagr']:>8.1f}%"
              f"{s['max_dd']:>8.1f}%{s['sharpe']:>8.2f}{s['blocked_circuit']:>9}{s['blocked_daily']:>8}")


def main():
    run_variant("A: original 40-symbol universe, ZERO added costs (reproduces the earlier sweep)",
                ORIGINAL_SYMBOLS, ORIGINAL_PCSE_SYMBOLS, cost_pct=0.0)

    run_variant(f"B: same 40-symbol universe, +{ROUND_TRIP_COST_PCT*100:.2f}% round-trip cost on breakout/swing",
                ORIGINAL_SYMBOLS, ORIGINAL_PCSE_SYMBOLS, cost_pct=ROUND_TRIP_COST_PCT)

    run_variant(f"C: expanded {len(EXPANDED_SYMBOLS)}-symbol universe (+{len(LAGGARD_AND_TURNAROUND_SYMBOLS)} "
                f"laggards/turnarounds, not survivorship-picked) + realistic costs",
                EXPANDED_SYMBOLS, EXPANDED_PCSE_SYMBOLS, cost_pct=ROUND_TRIP_COST_PCT)

    print(f"\n{'='*100}")
    print("Compare variant A's row to sweep_risk_levels.py's original output -- should match closely")
    print("(same universe, same cost=0). B isolates the cost effect. C isolates the universe effect.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
