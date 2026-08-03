"""Test adding a 200-day SMA trend filter to the long-only 1H+4H multi-timeframe
strategy: only take long entries when price is above its 200-day average.
Run on the combined in-sample + out-of-sample symbol universe (47 symbols) for
maximum statistical power, matching how the un-filtered baseline was evaluated.
"""
from backtest_mtf import run_backtest as run_backtest_mtf
from basket import CRYPTO_SYMBOLS, US_SYMBOLS, run_basket
from data import fetch_yfinance, fetch_yfinance_1h, fetch_yfinance_4h
from test_mtf_oos_symbols import OOS_CRYPTO_SYMBOLS, OOS_US_SYMBOLS

START = "2019-01-01"
END = "2026-08-01"

ALL_US = US_SYMBOLS + OOS_US_SYMBOLS
ALL_CRYPTO = CRYPTO_SYMBOLS + OOS_CRYPTO_SYMBOLS


def fetch_1h_4h_daily(symbol):
    return fetch_yfinance_1h(symbol), fetch_yfinance_4h(symbol), fetch_yfinance(symbol, START, END)


def main():
    print("=== 1H+4H long-only + 200-day SMA trend filter, combined 47-symbol universe ===")
    run_basket(
        fetch_fn=fetch_1h_4h_daily,
        backtest_fn=run_backtest_mtf,
        backtest_kwargs={"strategy_params": {"long_only": True}},  # trend_filter_df comes from fetch_fn's 3rd element
        symbols_by_group=(("US", ALL_US), ("Crypto", ALL_CRYPTO)),
    )


if __name__ == "__main__":
    main()
