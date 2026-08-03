"""CLI runner for the Probabilistic Candle-State Edge (PCSE) strategy: wires
alpaca_fetch's deep-history fetchers into basket.run_basket across the
US + crypto universe at a chosen timeframe.

Usage: python run_basket_candle_prob.py [1h|4h|daily] [long_only|both]
"""
import sys

import alpaca_fetch as af
import backtest_candle_prob as bt
import basket
from cache_fetch import cached_fetch

TIMEFRAME_FETCHERS = {
    "1h": (af.fetch_stock_1h, af.fetch_crypto_1h),
    "4h": (af.fetch_stock_4h, af.fetch_crypto_4h),
    "daily": (af.fetch_stock_daily, af.fetch_crypto_daily),
}


def make_fetch_fn(timeframe: str):
    stock_fn, crypto_fn = TIMEFRAME_FETCHERS[timeframe]

    def fetch_fn(symbol: str):
        if symbol in basket.CRYPTO_SYMBOLS:
            alpaca_symbol = symbol.replace("-", "/")  # Alpaca wants "BTC/USD", not yfinance's "BTC-USD"
            return cached_fetch(crypto_fn, alpaca_symbol, timeframe)
        return cached_fetch(stock_fn, symbol, timeframe)

    return fetch_fn


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    side_mode = sys.argv[2] if len(sys.argv) > 2 else "long_only"
    if timeframe not in TIMEFRAME_FETCHERS:
        raise SystemExit(f"timeframe must be one of {list(TIMEFRAME_FETCHERS)}, got {timeframe!r}")

    strategy_params = {"long_only": side_mode == "long_only"}
    print(f"=== PCSE basket backtest: timeframe={timeframe} long_only={strategy_params['long_only']} ===")

    trades, by_group, stats = basket.run_basket(
        fetch_fn=make_fetch_fn(timeframe),
        backtest_fn=bt.run_backtest,
        backtest_kwargs={"strategy_params": strategy_params},
    )
    return trades, by_group, stats


if __name__ == "__main__":
    main()
