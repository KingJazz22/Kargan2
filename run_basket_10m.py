"""Basket backtest on 10m bars (built from 5m, ~60 day Yahoo cap)."""
from basket import run_basket
from data import fetch_yfinance_10m


def main():
    run_basket(fetch_fn=fetch_yfinance_10m)


if __name__ == "__main__":
    main()
