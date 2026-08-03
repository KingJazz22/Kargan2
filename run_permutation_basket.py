"""CLI runner for PCSE's basket-wide random-entry permutation test: the
guard against candle-pattern-mining false discovery. Fits/predicts once per
symbol (reusing the same cache as run_basket_candle_prob.py), then compares
the real pooled t-stat against a null distribution built by re-running the
identical exit/sizing engine on randomly timed entries, pooled the same way
across the whole basket.

Usage: python run_permutation_basket.py [1h|4h|daily] [long_only|both] [n_shuffles]
"""
import sys

import backtest_candle_prob as bt
import basket
from run_basket_candle_prob import make_fetch_fn, TIMEFRAME_FETCHERS


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    side_mode = sys.argv[2] if len(sys.argv) > 2 else "long_only"
    n_shuffles = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    if timeframe not in TIMEFRAME_FETCHERS:
        raise SystemExit(f"timeframe must be one of {list(TIMEFRAME_FETCHERS)}, got {timeframe!r}")

    fetch_fn = make_fetch_fn(timeframe)
    dfs = {}
    for group, symbols in (("US", basket.US_SYMBOLS), ("Crypto", basket.CRYPTO_SYMBOLS)):
        for symbol in symbols:
            try:
                dfs[symbol] = fetch_fn(symbol)
            except Exception as e:
                print(f"  skip fetch {symbol}: {e}")

    strategy_params = {"long_only": side_mode == "long_only"}
    print(f"=== PCSE permutation test: timeframe={timeframe} long_only={strategy_params['long_only']} n_shuffles={n_shuffles} ===")
    result = bt.pooled_permutation_test(dfs, n_shuffles=n_shuffles, strategy_params=strategy_params)

    print("\nREAL pooled stats:")
    print(" long: ", result["real"]["long"])
    print(" short:", result["real"]["short"])
    print(f"\nnull percentile (real t-stat vs random-entry null), long:  {result['null_percentile_long']}")
    print(f"null percentile (real t-stat vs random-entry null), short: {result['null_percentile_short']}")
    print(f"\nskipped symbols: {result['skipped']}")
    return result


if __name__ == "__main__":
    main()
