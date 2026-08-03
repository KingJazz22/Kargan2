"""Basket backtest on native 30m bars. Yahoo caps this interval at ~60 days --
one narrow, recent market regime, thin sample for US stocks especially."""
from basket import run_basket
from data import fetch_yfinance_30m


def main():
    run_basket(fetch_fn=fetch_yfinance_30m)


if __name__ == "__main__":
    main()
