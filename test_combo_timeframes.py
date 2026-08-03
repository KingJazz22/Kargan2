"""Test the best 30m-sweep combo (RSI<25, Stoch<15, BB_std=2.0) across all
timeframes, to see if it holds up beyond the 30m window it was found on."""
from backtest_mr import run_backtest as run_backtest_mr
from basket import run_basket
from data import fetch_yfinance, fetch_yfinance_1h, fetch_yfinance_4h, fetch_yfinance_10m, fetch_yfinance_30m

START = "2019-01-01"
END = "2026-08-01"

STRATEGY_PARAMS = {
    "rsi_oversold": 25,
    "rsi_overbought": 75,
    "stoch_oversold": 15,
    "stoch_overbought": 85,
    "bb_std": 2.0,
    "kc_mult": 1.5,  # shown redundant with BB in the sweep, left at default
}

FETCHERS = {
    "daily": lambda symbol: fetch_yfinance(symbol, START, END),
    "4h": fetch_yfinance_4h,
    "1h": fetch_yfinance_1h,
    "30m": fetch_yfinance_30m,
    "10m": fetch_yfinance_10m,
}


def main():
    summary = []
    for label, fetch_fn in FETCHERS.items():
        print(f"\n=== Combo (RSI<25, Stoch<15, BB_std=2.0) on {label} bars ===")
        _, _, stats = run_basket(
            fetch_fn=fetch_fn,
            backtest_fn=run_backtest_mr,
            backtest_kwargs={"strategy_params": STRATEGY_PARAMS},
            verbose=True,
        )
        stats["timeframe"] = label
        summary.append(stats)

    print("\n\n=== Summary across timeframes ===")
    import pandas as pd
    df = pd.DataFrame(summary).set_index("timeframe")
    cols = ["num_trades", "win_rate_pct", "avg_r_multiple", "profit_factor", "t_stat_vs_zero"]
    print(df[cols].to_string())


if __name__ == "__main__":
    main()
