"""Mean Reversion basket backtest, across timeframes. Usage:
    python run_basket_meanrev.py [daily|4h|1h|30m|10m]
"""
import sys

from backtest_mr import run_backtest as run_backtest_mr
from basket import run_basket
from data import fetch_yfinance, fetch_yfinance_1h, fetch_yfinance_4h, fetch_yfinance_10m, fetch_yfinance_30m

START = "2019-01-01"
END = "2026-08-01"

FETCHERS = {
    "daily": lambda symbol: fetch_yfinance(symbol, START, END),
    "4h": fetch_yfinance_4h,
    "1h": fetch_yfinance_1h,
    "30m": fetch_yfinance_30m,
    "10m": fetch_yfinance_10m,
}


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if timeframe not in FETCHERS:
        raise SystemExit(f"unknown timeframe {timeframe!r}; choose from {list(FETCHERS)}")
    print(f"=== Mean Reversion, {timeframe} bars ===")
    run_basket(fetch_fn=FETCHERS[timeframe], backtest_fn=run_backtest_mr)


if __name__ == "__main__":
    main()
