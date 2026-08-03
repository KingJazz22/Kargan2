"""Parameter sweep: TP_ATR_MULT, long-only, evaluated on IN-SAMPLE and
OUT-OF-SAMPLE together for every candidate -- not tuned on in-sample first
and checked afterward. The whole reason for this sweep is that the previous
tweak (MIN_BARS_BETWEEN_LEGS) only ever moved in-sample, because take-profit
(0.75x ATR) sits closer than the leg-add trigger (1.0x ATR) and resolves most
trades before the grid ever gets a chance to average down -- in EITHER
sample. Widening TP relative to the leg-step should let the averaging
mechanic actually engage on out-of-sample symbols too, which is the real
test of whether it does anything.

Data is fetched ONCE per symbol (raw OHLCV, prepare() is re-run per TP value
inside run_backtest, which is cheap -- the network fetch is what's slow).
"""
import numpy as np

import backtest_grid_martingale as bt
from basket import US_SYMBOLS
from bayesian_tracker import BetaBinomialPosterior
from data import fetch_yfinance
from test_mtf_oos_symbols import OOS_US_SYMBOLS

START = "2019-01-01"
END = "2026-08-01"
TP_SWEEP = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def prefetch(symbols):
    out = {}
    for symbol in symbols:
        try:
            out[symbol] = fetch_yfinance(symbol, START, END)
        except Exception as e:
            print(f"  skip {symbol}: {e}")
    return out


def run_group(dfs, tp_atr_mult):
    posterior = BetaBinomialPosterior()
    all_trades = []
    for symbol, df in dfs.items():
        try:
            trades, _ = bt.run_backtest(df, posterior=posterior, tp_atr_mult=tp_atr_mult)
        except ValueError:
            continue
        closed = [t for t in trades if t.exit_price is not None]
        all_trades.extend(closed)
    return all_trades


def stats(trades):
    if not trades:
        return {"n": 0, "win_rate": None, "avg_R": None, "t": None, "multi_leg_pct": None}
    r = np.array([t.r_multiple() for t in trades])
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    t_stat = r.mean() / se if se > 0 else 0.0
    multi_leg_pct = 100 * sum(1 for t in trades if t.num_legs >= 2) / len(trades)
    return {
        "n": len(trades),
        "win_rate": round(100 * np.mean(r > 0), 1),
        "avg_R": round(float(r.mean()), 3),
        "t": round(float(t_stat), 2),
        "multi_leg_pct": round(multi_leg_pct, 1),
    }


def main():
    print("Fetching in-sample symbols...")
    is_dfs = prefetch(US_SYMBOLS)
    print("Fetching out-of-sample symbols...")
    oos_dfs = prefetch(OOS_US_SYMBOLS)

    header = f"{'TP':>5}  {'IS n':>5}  {'IS win%':>8}  {'IS avgR':>8}  {'IS t':>6}  {'IS ml%':>6}   " \
             f"{'OOS n':>5}  {'OOS win%':>9}  {'OOS avgR':>9}  {'OOS t':>6}  {'OOS ml%':>7}   " \
             f"{'Comb t':>7}"
    print(header)
    print("-" * len(header))

    for tp in TP_SWEEP:
        is_trades = run_group(is_dfs, tp)
        oos_trades = run_group(oos_dfs, tp)
        s_is = stats(is_trades)
        s_oos = stats(oos_trades)
        s_comb = stats(is_trades + oos_trades)
        print(f"{tp:>5.2f}  {s_is['n']:>5}  {s_is['win_rate']:>8}  {s_is['avg_R']:>8}  {s_is['t']:>6}  {s_is['multi_leg_pct']:>6}   "
              f"{s_oos['n']:>5}  {s_oos['win_rate']:>9}  {s_oos['avg_R']:>9}  {s_oos['t']:>6}  {s_oos['multi_leg_pct']:>7}   "
              f"{s_comb['t']:>7}")


if __name__ == "__main__":
    main()
