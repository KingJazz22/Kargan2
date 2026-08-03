"""Out-of-sample check: does the Grid-Martingale result on the original
20-symbol US basket hold on a completely different set of US symbols never
used anywhere else in this project (test_mtf_oos_symbols.OOS_US_SYMBOLS)?
Same in-sample/out-of-sample split this project already uses elsewhere --
US only, per the v1 scope decision (see FINDINGS_GRID_MARTINGALE.md).
"""
import numpy as np

import backtest_grid_martingale as bt
from basket import US_SYMBOLS, run_basket
from bayesian_tracker import BetaBinomialPosterior
from data import fetch_yfinance
from test_mtf_oos_symbols import OOS_US_SYMBOLS

START = "2019-01-01"
END = "2026-08-01"


def _stats(trades):
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return {"num_trades": 0}
    r = np.array([t.r_multiple() for t in closed])
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    t_stat = r.mean() / se if se > 0 else 0.0
    return {
        "num_trades": len(closed),
        "win_rate_pct": round(100 * np.mean(r > 0), 1),
        "avg_r_multiple": round(float(r.mean()), 3),
        "t_stat_vs_zero": round(float(t_stat), 2),
    }


def main():
    fetch_fn = lambda symbol: fetch_yfinance(symbol, START, END)

    print("=== In-sample (original 20 US symbols) ===")
    in_sample_trades, _, _ = run_basket(
        fetch_fn=fetch_fn, symbols_by_group=(("US", US_SYMBOLS),),
        backtest_fn=bt.run_backtest, backtest_kwargs={"posterior": BetaBinomialPosterior()},
    )

    print("\n=== Out-of-sample (20 fresh US symbols, never used elsewhere) ===")
    oos_trades, _, _ = run_basket(
        fetch_fn=fetch_fn, symbols_by_group=(("US", OOS_US_SYMBOLS),),
        backtest_fn=bt.run_backtest, backtest_kwargs={"posterior": BetaBinomialPosterior()},
    )

    combined = in_sample_trades + oos_trades

    print("\n=== Summary ===")
    print(f"  In-sample:     {_stats(in_sample_trades)}")
    print(f"  Out-of-sample: {_stats(oos_trades)}")
    print(f"  Combined:      {_stats(combined)}")

    for label, trades in (("In-sample", in_sample_trades), ("Out-of-sample", oos_trades), ("Combined", combined)):
        for side in ("long", "short"):
            side_trades = [t for t in trades if t.side == side]
            print(f"  {label} / {side}: {_stats(side_trades)}")


if __name__ == "__main__":
    main()
