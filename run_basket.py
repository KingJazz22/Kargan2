"""Basket backtest on DAILY bars: unchanged strategy across many symbols,
trades pooled for a statistically meaningful sample. See FINDINGS.md."""
from basket import run_basket
from data import fetch_yfinance

START = "2019-01-01"
END = "2026-08-01"


def main():
    run_basket(fetch_fn=lambda symbol: fetch_yfinance(symbol, START, END))


if __name__ == "__main__":
    main()
