"""Basket backtest on native 1H bars. Same ~2-year Yahoo hourly-data cap as
the 4H run, but ~4x more bars per symbol -- faster warmup burn, more setups,
more noise sensitivity in the pullback/candle rules."""
from basket import run_basket
from data import fetch_yfinance_1h


def main():
    run_basket(fetch_fn=fetch_yfinance_1h)


if __name__ == "__main__":
    main()
