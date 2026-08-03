"""Parameter sweep: mean-reversion entry thresholds on 30m bars.

Fetches 30m OHLCV once per symbol (Yahoo network calls), then re-runs the
backtest against the cached data for every threshold combination -- only
indicator/signal recomputation, no re-fetching.

Sweeps RSI oversold level, Stochastic oversold level, Bollinger stdev, and
Keltner ATR multiplier. Overbought/short thresholds mirror the oversold/long
ones (100 - level), matching the spec's "short is exactly opposite."
"""
import itertools

import numpy as np
import pandas as pd

from backtest_mr import run_backtest as run_backtest_mr
from basket import CRYPTO_SYMBOLS, US_SYMBOLS, run_basket
from data import fetch_yfinance_30m

RSI_OVERSOLD_GRID = [20, 25, 30, 35]
STOCH_OVERSOLD_GRID = [15, 20, 25]
BB_STD_GRID = [1.5, 2.0, 2.5]
KC_MULT_GRID = [1.0, 1.5, 2.0]


def build_cache():
    cache = {}
    for symbol in US_SYMBOLS + CRYPTO_SYMBOLS:
        try:
            cache[symbol] = fetch_yfinance_30m(symbol)
        except Exception as e:
            print(f"  skip fetch {symbol}: {e}")
    return cache


def main():
    print("Fetching 30m data once for all symbols...")
    cache = build_cache()
    print(f"Cached {len(cache)} symbols.\n")

    def cached_fetch(symbol):
        return cache[symbol]

    results = []
    combos = list(itertools.product(RSI_OVERSOLD_GRID, STOCH_OVERSOLD_GRID, BB_STD_GRID, KC_MULT_GRID))
    print(f"Running {len(combos)} parameter combinations...\n")

    for rsi_os, stoch_os, bb_std, kc_mult in combos:
        strategy_params = {
            "rsi_oversold": rsi_os,
            "rsi_overbought": 100 - rsi_os,
            "stoch_oversold": stoch_os,
            "stoch_overbought": 100 - stoch_os,
            "bb_std": bb_std,
            "kc_mult": kc_mult,
        }
        _, _, stats = run_basket(
            fetch_fn=cached_fetch,
            backtest_fn=run_backtest_mr,
            backtest_kwargs={"strategy_params": strategy_params},
            verbose=False,
        )
        stats.update({"rsi_os": rsi_os, "stoch_os": stoch_os, "bb_std": bb_std, "kc_mult": kc_mult})
        results.append(stats)

    df = pd.DataFrame(results)
    df.to_csv("sweep_meanrev_30m_results.csv", index=False)

    valid = df[df["num_trades"] >= 20].copy()
    print(f"Combos with >=20 pooled trades: {len(valid)} / {len(df)}\n")

    print("=== Top 10 by avg_r_multiple (min 20 trades) ===")
    print(valid.sort_values("avg_r_multiple", ascending=False).head(10)[
        ["rsi_os", "stoch_os", "bb_std", "kc_mult", "num_trades", "win_rate_pct", "avg_r_multiple", "profit_factor", "t_stat_vs_zero"]
    ].to_string(index=False))

    print("\n=== Bottom 10 by avg_r_multiple (min 20 trades) ===")
    print(valid.sort_values("avg_r_multiple", ascending=True).head(10)[
        ["rsi_os", "stoch_os", "bb_std", "kc_mult", "num_trades", "win_rate_pct", "avg_r_multiple", "profit_factor", "t_stat_vs_zero"]
    ].to_string(index=False))

    print("\n=== Distribution of avg_r_multiple across all valid combos ===")
    print(valid["avg_r_multiple"].describe().to_string())
    print(f"\nFraction of combos with avg_r_multiple > 0: {round(100*(valid['avg_r_multiple']>0).mean(),1)}%")
    print(f"Fraction of combos with t_stat_vs_zero > 2: {round(100*(valid['t_stat_vs_zero']>2).mean(),1)}%")

    print(f"\nFull results saved -> sweep_meanrev_30m_results.csv")


if __name__ == "__main__":
    main()
