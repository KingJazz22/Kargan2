"""Out-of-sample check: does the long-only 1H+4H multi-timeframe RSI+Stochastic
edge (t=2.72 on the original 28-symbol basket) hold on a completely different
set of symbols never used anywhere else in this project?
"""
from backtest_mtf import run_backtest as run_backtest_mtf
from basket import run_basket
from data import fetch_yfinance_1h, fetch_yfinance_4h

# Different sectors/caps than the original US basket (mega-cap tech-heavy),
# and different coins than the original crypto basket (majors by market cap).
OOS_US_SYMBOLS = [
    "IWM", "DIA", "KO", "PEP", "PG", "MRK", "PFE", "BAC", "WFC", "GS",
    "CAT", "BA", "COST", "MCD", "NKE", "INTC", "CSCO", "ORCL", "ADBE", "QCOM",
]
OOS_CRYPTO_SYMBOLS = [
    "LTC-USD", "LINK-USD", "DOT-USD", "UNI-USD", "ATOM-USD", "ETC-USD", "BCH-USD", "XLM-USD",
]


def fetch_1h_4h(symbol):
    return fetch_yfinance_1h(symbol), fetch_yfinance_4h(symbol)


def main():
    print("=== 1H+4H long-only, OUT-OF-SAMPLE symbols ===")
    run_basket(
        fetch_fn=fetch_1h_4h,
        backtest_fn=run_backtest_mtf,
        backtest_kwargs={"strategy_params": {"long_only": True}},
        symbols_by_group=(("US", OOS_US_SYMBOLS), ("Crypto", OOS_CRYPTO_SYMBOLS)),
    )


if __name__ == "__main__":
    main()
