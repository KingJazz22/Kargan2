"""CLI runner for PCSE's final validated configuration: wires
alpaca_fetch's deep-history fetchers into basket.run_basket across the
US + crypto universe. This is the strategy to point at a paper account,
not strategy_candle_prob/backtest_candle_prob (superseded, see
FINDINGS_CANDLE_PROB.md).

Usage: python run_basket_pcse_final.py [1h|4h|daily]
"""
import sys

import backtest_pcse_final as bt
import basket
from run_basket_candle_prob import make_fetch_fn, TIMEFRAME_FETCHERS


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    if timeframe not in TIMEFRAME_FETCHERS:
        raise SystemExit(f"timeframe must be one of {list(TIMEFRAME_FETCHERS)}, got {timeframe!r}")

    print(f"=== PCSE final config basket backtest: timeframe={timeframe} ===")
    trades, by_group, stats = basket.run_basket(
        fetch_fn=make_fetch_fn(timeframe),
        backtest_fn=bt.run_backtest,
    )
    return trades, by_group, stats


if __name__ == "__main__":
    main()
