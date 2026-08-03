"""Multi-Timeframe RSI+Stochastic basket backtest: LTF=1H execution, HTF=4H
confirmation. Both oversold (or both overbought) simultaneously -> entry."""
from backtest_mtf import run_backtest as run_backtest_mtf
from basket import run_basket
from data import fetch_yfinance_1h, fetch_yfinance_4h


def fetch_1h_4h(symbol):
    return fetch_yfinance_1h(symbol), fetch_yfinance_4h(symbol)


def main():
    run_basket(fetch_fn=fetch_1h_4h, backtest_fn=run_backtest_mtf)


if __name__ == "__main__":
    main()
