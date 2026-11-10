"""Compare Breakout Hunter baseline vs. VXX-filtered variants, same universe
and dates as FINDINGS_BREAKOUT.md (US only -- that study found no crypto edge
at any timeframe, and VXX has no crypto-equivalent meaning here).

Usage: python run_basket_breakout_vxx.py
"""
import numpy as np
import pandas as pd

from backtest_breakout_vxx import run_backtest
from basket import US_SYMBOLS, STARTING_EQUITY, RISK_PCT
from data import fetch_yfinance
from test_mtf_oos_symbols import OOS_US_SYMBOLS

START = "2019-01-01"
END = "2026-08-01"

VARIANTS = [
    "long_entry",              # baseline, unfiltered
    "long_entry_calm",
    "long_entry_falling",
    "long_entry_not_spiking",
    "long_entry_calm_falling",
    "long_entry_all3",
]


def pooled_stats(trades):
    if not trades:
        return {"num_trades": 0}
    r = np.array([t.r_multiple() for t in trades])
    pnls = np.array([t.pnl() for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    return {
        "num_trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 1),
        "avg_r": round(float(r.mean()), 3),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 2) if losses.sum() != 0 else float("inf"),
        "t_stat": round(float(r.mean() / se), 2) if se > 0 else 0.0,
    }


def main():
    vxx = fetch_yfinance("VXX", START, END)
    print(f"VXX bars fetched: {len(vxx)} ({vxx.index[0].date()} -> {vxx.index[-1].date()})\n")

    symbol_groups = [("IS", US_SYMBOLS), ("OOS", OOS_US_SYMBOLS)]
    results = {v: {"IS": [], "OOS": []} for v in VARIANTS}

    for group_name, symbols in symbol_groups:
        for symbol in symbols:
            try:
                df = fetch_yfinance(symbol, START, END)
            except Exception as e:
                print(f"  skip {symbol}: {e}")
                continue
            for variant in VARIANTS:
                try:
                    trades, _ = run_backtest(df, vxx, variant, starting_equity=STARTING_EQUITY, risk_pct=RISK_PCT)
                except Exception as e:
                    print(f"  skip {symbol}/{variant}: {e}")
                    continue
                results[variant][group_name].extend([t for t in trades if t.exit_price is not None])
            print(f"  fetched+tested {group_name:4s} {symbol}")

    rows = []
    for variant in VARIANTS:
        is_trades = results[variant]["IS"]
        oos_trades = results[variant]["OOS"]
        combined = is_trades + oos_trades
        is_s, oos_s, comb_s = pooled_stats(is_trades), pooled_stats(oos_trades), pooled_stats(combined)
        rows.append({
            "variant": variant,
            "IS_trades": is_s.get("num_trades", 0), "IS_avg_R": is_s.get("avg_r"), "IS_t": is_s.get("t_stat"),
            "OOS_trades": oos_s.get("num_trades", 0), "OOS_avg_R": oos_s.get("avg_r"), "OOS_t": oos_s.get("t_stat"),
            "Combined_trades": comb_s.get("num_trades", 0), "Combined_avg_R": comb_s.get("avg_r"),
            "Combined_t": comb_s.get("t_stat"), "Combined_win%": comb_s.get("win_rate_pct"),
            "Combined_PF": comb_s.get("profit_factor"),
        })

    df_out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== Breakout Hunter: baseline vs. VXX-filtered variants (US only) ===")
    print(df_out.to_string(index=False))
    df_out.to_csv("breakout_vxx_results.csv", index=False)
    print("\nSaved to breakout_vxx_results.csv")


if __name__ == "__main__":
    main()
