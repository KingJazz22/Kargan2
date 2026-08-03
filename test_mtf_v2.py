"""MTF v2: TSL-only exit (no RSI-neutral profit target), tested on two new
timeframe pairs not yet tried -- 4H+Daily and 30m+1H -- alongside the
original validated 1H+4H pairing for comparison. Long-only, same 40-symbol
validated US universe throughout.
"""
from backtest_mtf import run_backtest as run_backtest_mtf
from basket import US_SYMBOLS, run_basket
from data import fetch_yfinance, fetch_yfinance_1h, fetch_yfinance_30m, fetch_yfinance_4h
from test_mtf_oos_symbols import OOS_US_SYMBOLS

START = "2019-01-01"
END = "2026-08-01"
ALL_US = US_SYMBOLS + OOS_US_SYMBOLS


def fetch_1h_4h(symbol):
    return fetch_yfinance_1h(symbol), fetch_yfinance_4h(symbol)


def fetch_4h_daily(symbol):
    return fetch_yfinance_4h(symbol), fetch_yfinance(symbol, START, END)


def fetch_30m_1h(symbol):
    return fetch_yfinance_30m(symbol), fetch_yfinance_1h(symbol)


CONFIGS = {
    "1H+4H (original, v1 exit for reference)": (fetch_1h_4h, {"long_only": True, "htf_bar_hours": 4}),
    "1H+4H (TSL-only exit)": (fetch_1h_4h, {"long_only": True, "htf_bar_hours": 4, "tsl_only_exit": True}),
    "4H+Daily (TSL-only exit)": (fetch_4h_daily, {"long_only": True, "htf_bar_hours": 24, "tsl_only_exit": True}),
    "30m+1H (TSL-only exit)": (fetch_30m_1h, {"long_only": True, "htf_bar_hours": 1, "tsl_only_exit": True}),
}


def main():
    for label, (fetch_fn, params) in CONFIGS.items():
        print(f"\n=== {label} ===")
        run_basket(
            fetch_fn=fetch_fn,
            backtest_fn=run_backtest_mtf,
            backtest_kwargs={"strategy_params": params},
            symbols_by_group=(("US", ALL_US),),
        )


if __name__ == "__main__":
    main()
