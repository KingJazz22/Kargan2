"""Ground truth: re-run the ORIGINAL event-driven engine (backtest_swing_breakout.py)
against the pickled cache (fetch_cache_swing_breakout.py) instead of live yfinance,
so results are reproducible and yfinance non-determinism can't confound the
vectorbt-port comparison. This run's saved output IS the fixed target going
forward -- not FINDINGS_SWING_STRUCTURE.md's numbers, which used live data at
an earlier point in time on a possibly different date range.

Usage: .venv/Scripts/python.exe run_original_cached_swing_breakout.py
"""
import pickle

from backtest_swing_breakout import run_backtest as run_backtest_swing_breakout
from basket import US_SYMBOLS, run_basket
from fetch_cache_swing_breakout import load_cached
from test_mtf_oos_symbols import OOS_US_SYMBOLS


def trade_to_dict(t):
    return {
        "symbol": getattr(t, "symbol", None),
        "group": getattr(t, "group", None),
        "side": t.side,
        "entry_date": t.entry_date,
        "entry_price": t.entry_price,
        "shares": t.shares,
        "initial_stop": t.initial_stop,
        "emergency_stop": t.emergency_stop,
        "atr_at_entry": t.atr_at_entry,
        "risk_amount": t.risk_amount,
        "exit_date": t.exit_date,
        "exit_price": t.exit_price,
        "exit_reason": t.exit_reason,
        "pnl": t.pnl(),
        "r_multiple": t.r_multiple(),
    }


def run_and_save(group_name, symbols, out_path):
    print(f"\n=== Swing-structure breakout (ORIGINAL engine, cached data), {group_name} ===")
    all_trades, by_group, stats = run_basket(
        fetch_fn=load_cached,
        backtest_fn=run_backtest_swing_breakout,
        symbols_by_group=((group_name, symbols),),
    )
    payload = {
        "group_name": group_name,
        "stats": stats,
        "trades": [trade_to_dict(t) for t in all_trades],
        "per_symbol": by_group,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {len(all_trades)} trades + stats to {out_path}")
    return stats


def main():
    is_stats = run_and_save("US", US_SYMBOLS, "swing_breakout_original_IS.pkl")
    oos_stats = run_and_save("US_OOS", OOS_US_SYMBOLS, "swing_breakout_original_OOS.pkl")

    print("\n\n=== SUMMARY (original engine, cached data) ===")
    print(f"IS : {is_stats}")
    print(f"OOS: {oos_stats}")


if __name__ == "__main__":
    main()
