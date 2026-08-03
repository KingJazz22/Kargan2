"""Basket backtest on 4H bars: unchanged strategy across many symbols.
Yahoo caps hourly history at ~730 days, so this covers ~2 years, not the
~6.5 years the daily basket used -- expect a smaller trade sample."""
from basket import run_basket
from data import fetch_yfinance_4h


def main():
    run_basket(fetch_fn=fetch_yfinance_4h)


if __name__ == "__main__":
    main()
