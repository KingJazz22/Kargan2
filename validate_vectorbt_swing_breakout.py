"""Validate the vectorbt port (run_vectorbt_swing_breakout.py) against the
original engine's cached-data ground truth (run_original_cached_swing_breakout.py).

Tier 1 (per-trade, SPY/AAPL/NVDA): join by nearest entry_date, diff
entry_price/exit_date/exit_price/exit_reason, print mismatches with
surrounding OHLC so a human can see whether it's the known/expected
same-bar-entry-exit / emergency-stop / entry-day-spike divergence or a bug.

Tier 2 (pooled stats, full IS+OOS baskets): delta table with explicit
PASS/FAIL against stated tolerances.
"""
import pickle

import numpy as np
import pandas as pd

from fetch_cache_swing_breakout import load_cached

TIER1_SYMBOLS = ["SPY", "AAPL", "NVDA"]

TOLERANCES = {
    "num_trades": 2,
    "win_rate_pct": 2.0,
    "avg_r_multiple": 0.02,
    "t_stat_vs_zero": 0.3,
    "profit_factor": 0.15,
}


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def trades_to_df(payload):
    if isinstance(payload["trades"], pd.DataFrame):
        return payload["trades"]
    return pd.DataFrame(payload["trades"])


def tier1_symbol(symbol, orig_df, vbt_df):
    o = orig_df[orig_df["symbol"] == symbol].sort_values("entry_date").reset_index(drop=True)
    v = vbt_df[vbt_df["symbol"] == symbol].sort_values("entry_date").reset_index(drop=True)
    ohlc = load_cached(symbol)

    print(f"\n--- Tier 1: {symbol}  (original n={len(o)}, vbt n={len(v)}) ---")

    used_v = set()
    rows = []
    n_match = 0
    n_total = 0
    for _, orow in o.iterrows():
        n_total += 1
        # nearest entry_date in v not yet used
        if len(v) == 0:
            rows.append((orow, None))
            continue
        deltas = (v["entry_date"] - orow["entry_date"]).abs()
        deltas_avail = deltas.copy()
        if used_v:
            deltas_avail.loc[list(used_v)] = pd.Timedelta(days=99999)
        best_idx = deltas_avail.idxmin()
        if deltas_avail[best_idx] > pd.Timedelta(days=5):
            rows.append((orow, None))
            continue
        used_v.add(best_idx)
        vrow = v.loc[best_idx]
        rows.append((orow, vrow))

    print(f"{'entry_date':12s} {'o.entry':>9s} {'v.entry':>9s}  {'o.exit_date':12s} {'v.exit_date':12s} "
          f"{'o.exit':>9s} {'v.exit':>9s}  {'o.reason':13s} {'v.reason':13s}  match")
    for orow, vrow in rows:
        if vrow is None:
            print(f"{str(orow['entry_date'].date()):12s} {orow['entry_price']:9.2f} {'--':>9s}  "
                  f"{str(orow['exit_date'].date()):12s} {'--':12s} {orow['exit_price']:9.2f} {'--':>9s}  "
                  f"{orow['exit_reason']:13s} {'--':13s}  NO_MATCH")
            continue
        price_match = abs(orow["entry_price"] - vrow["entry_price"]) < 0.05 and abs(orow["exit_price"] - vrow["exit_price"]) < 0.05
        date_match = orow["exit_date"] == vrow["exit_date"]
        match = price_match and date_match
        n_match += int(match)
        flag = "OK" if match else "MISMATCH"
        print(f"{str(orow['entry_date'].date()):12s} {orow['entry_price']:9.2f} {vrow['entry_price']:9.2f}  "
              f"{str(orow['exit_date'].date()):12s} {str(vrow['exit_date'].date()):12s} "
              f"{orow['exit_price']:9.2f} {vrow['exit_price']:9.2f}  "
              f"{orow['exit_reason']:13s} {vrow['exit_reason']:13s}  {flag}")
        if not match:
            print(f"    --- surrounding OHLC for {symbol}, entry {orow['entry_date'].date()} -> "
                  f"orig exit {orow['exit_date'].date()} / vbt exit {vrow['exit_date'].date()} ---")
            lo = min(orow["entry_date"], vrow["entry_date"]) - pd.Timedelta(days=2)
            hi = max(orow["exit_date"], vrow["exit_date"]) + pd.Timedelta(days=2)
            window = ohlc.loc[(ohlc.index >= lo) & (ohlc.index <= hi)]
            print(window[["Open", "High", "Low", "Close"]].round(2).to_string())

    frac = n_match / n_total if n_total else 0.0
    print(f"\n{symbol}: {n_match}/{n_total} trades match within tolerance ({frac*100:.1f}%)")
    return n_match, n_total


def tier2_group(name, orig_stats, vbt_stats):
    print(f"\n=== Tier 2: {name} ===")
    print(f"{'metric':20s} {'original':>12s} {'vectorbt':>12s} {'delta':>10s} {'tol':>8s}  status")
    all_pass = True
    for metric, tol in TOLERANCES.items():
        ov = orig_stats.get(metric)
        vv = vbt_stats.get(metric)
        if ov is None or vv is None:
            print(f"{metric:20s} {'--':>12s} {'--':>12s} {'--':>10s} {'--':>8s}  SKIP (missing)")
            continue
        delta = vv - ov
        ok = abs(delta) <= tol
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"{metric:20s} {ov:12.3f} {vv:12.3f} {delta:10.3f} {tol:8.3f}  {status}")
    print(f"{name} overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main():
    orig_is = load("swing_breakout_original_IS.pkl")
    orig_oos = load("swing_breakout_original_OOS.pkl")
    vbt_is = load("swing_breakout_vectorbt_IS.pkl")
    vbt_oos = load("swing_breakout_vectorbt_OOS.pkl")

    orig_is_df = trades_to_df(orig_is)
    orig_oos_df = trades_to_df(orig_oos)
    vbt_is_df = trades_to_df(vbt_is)
    vbt_oos_df = trades_to_df(vbt_oos)

    print("=" * 78)
    print("TIER 1: per-trade diff on SPY / AAPL / NVDA (in-sample basket)")
    print("=" * 78)
    total_match, total_n = 0, 0
    for sym in TIER1_SYMBOLS:
        m, n = tier1_symbol(sym, orig_is_df, vbt_is_df)
        total_match += m
        total_n += n
    print(f"\nTIER 1 OVERALL: {total_match}/{total_n} = {100*total_match/total_n:.1f}% match "
          f"(target >=95%, with mismatches traceable to documented divergences)")

    print("\n" + "=" * 78)
    print("TIER 2: pooled stats, IS and OOS")
    print("=" * 78)
    is_pass = tier2_group("IS (US)", orig_is["stats"], vbt_is["stats"])
    oos_pass = tier2_group("OOS (US_OOS)", orig_oos["stats"], vbt_oos["stats"])

    print("\n" + "=" * 78)
    print(f"FINAL: Tier 2 IS {'PASS' if is_pass else 'FAIL'} / OOS {'PASS' if oos_pass else 'FAIL'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
