"""Random-entry benchmark for the other live strategies (FINDINGS_RANDOM_ENTRY.md
did swing_pcse_exit): does each strategy's ENTRY beat random entries run
through its own, unchanged exit engine?

  breakout_hunter  backtest_breakout.py        daily, S&P point-in-time (v1)
  swing_structure  backtest_swing_breakout.py  daily, S&P point-in-time (v1)
  pcse             backtest_pcse_final.py      4H, the 20 PCSE symbols (v1 + v2)

Each engine's `strat` module is swapped for a wrapper whose prepare() returns
the real prepared frame with only `long_entry` replaced: the real signals,
or the same number of signals on uniformly random allowed days (past the
engine warmup, index member for fresh names, entry inputs finite). Stops,
trails, targets and sizing all come from the real code. Real-entry trades
are saved to _real_trades_all_bots.pkl for compare_bots_buy_hold.py.
Writes research_metalabel/results_random_entry_all_bots.txt.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_breakout as bt_bo
import backtest_metalabel_pcse as ml
import backtest_pcse_final as bt_pcse
import backtest_pcse_fresh_universe as fu
import backtest_swing_breakout as bt_sw
import strategy_breakout as strat_bo
import strategy_pcse_final as strat_pcse
import strategy_swing_breakout as strat_sw
from basket import US_SYMBOLS

N_DRAWS = 10
SAVE = "_real_trades_all_bots.pkl"


class EntrySwap:
    """Stands in for a strategy module inside an engine: attribute lookups go to the
    real module; prepare() hands back a pre-computed frame with chosen entries."""

    def __init__(self, real):
        self._real, self.frame = real, None

    def __getattr__(self, name):
        return getattr(self._real, name)

    def prepare(self, df, **kwargs):
        return self.frame


STRATEGIES = {
    "breakout_hunter": dict(engine=bt_bo, real=strat_bo, need=["support", "atr"], data="daily"),
    "swing_structure": dict(engine=bt_sw, real=strat_sw, need=["support", "atr"], data="daily"),
    "pcse": dict(engine=bt_pcse, real=strat_pcse, need=["ret_stdev"], data="4h"),
}


def daily_segments():
    d = fu.dedupe(fu.load_bars())
    spy = d[d["symbol"] == "SPY"].set_index("date")
    cal_pos = pd.Series(np.arange(len(spy.index)), index=spy.index)
    out = []
    for sym, g in d.groupby("symbol"):
        if sym == "SPY":
            continue
        g = g[g["date"].isin(spy.index)]
        if not np.isfinite(g["Close"]).all():
            continue
        for _, seg in g.groupby(fu.split_segments(g, cal_pos)):
            if len(seg) < 400:
                continue
            s = seg.set_index("date")
            mask = pd.Series(True, index=s.index) if sym in fu.OLD_SET else s["member"].astype(bool)
            out.append((sym, s[["Open", "High", "Low", "Close", "Volume"]].astype(float), mask))
    return out


def fourh_segments():
    c = pickle.load(open("_sweep_risk_levels_bars_cache.pkl", "rb"))
    return [(s, c[("4h", s)].astype(float), pd.Series(True, index=c[("4h", s)].index))
            for s in US_SYMBOLS if ("4h", s) in c]


def trade_rows(sym, trades, kind):
    rows = []
    for t in trades:
        if t.exit_price is None:
            continue
        stop = getattr(t, "stop_distance", None)
        if stop is None:
            stop = t.entry_price - t.initial_stop
        rows.append({"strategy": kind, "symbol": sym, "entry_time": t.entry_date, "exit_time": t.exit_date,
                     "entry_price": t.entry_price, "exit_price": t.exit_price, "stop_distance": stop,
                     "R": t.r_multiple(), "exit_reason": t.exit_reason})
    return rows


def run_strategy(name, cfg, segs, say, rng):
    eng, real = cfg["engine"], cfg["real"]
    swap = EntrySwap(real)
    eng.strat = swap
    prepped = []
    for sym, df, mask in segs:
        try:
            p = real.prepare(df)
        except Exception:
            continue
        if len(p) <= eng.WARMUP_BARS + 5:
            continue
        m = mask.reindex(p.index).fillna(False).to_numpy(bool)
        ok = m.copy()
        ok[:eng.WARMUP_BARS] = False
        ok[-1] = False
        for col in cfg["need"]:
            v = p[col].to_numpy(float)
            ok &= np.isfinite(v) & (v > 0 if col != "support" else True)
        real_sig = p["long_entry"].to_numpy(bool) & m
        prepped.append((sym, p, real_sig, np.flatnonzero(ok), int(real_sig[eng.WARMUP_BARS:].sum())))

    def run(choose):
        rows = []
        for sym, p, real_sig, allowed, n_sig in prepped:
            f = p.copy()
            f["long_entry"] = choose(real_sig, allowed, n_sig, len(p))
            swap.frame = f
            trades, _ = eng.run_backtest(f, starting_equity=1e12, risk_pct=0.01)
            rows += trade_rows(sym, trades, name)
        t = pd.DataFrame(rows)
        hold = (pd.to_datetime(t["exit_time"]) - pd.to_datetime(t["entry_time"])).dt.total_seconds() / 86400
        R = t["R"].to_numpy()
        return t, {"trades": len(t), "win%": (R > 0).mean() * 100, "avg_R": R.mean(),
                   "t": R.mean() / R.std(ddof=1) * np.sqrt(len(R)), "avg_hold_d": hold.mean(),
                   "R_per_100d": R.sum() / hold.clip(lower=1 / 6.5).sum() * 100}

    def random_choice(real_sig, allowed, n_sig, n):
        out = np.zeros(n, dtype=bool)
        if n_sig and len(allowed):
            out[rng.choice(allowed, size=min(n_sig, len(allowed)), replace=False)] = True
        return out

    real_trades, real_stats = run(lambda real_sig, allowed, n_sig, n: real_sig)
    say(f"\n=== {name}: {len(prepped)} segments ===")
    say(f"  REAL entries : " + " ".join(f"{k}={v:.3f}" for k, v in real_stats.items()))
    draws = []
    for b in range(N_DRAWS):
        _, st = run(random_choice)
        draws.append(st)
        say(f"  random {b + 1:2d}    : " + " ".join(f"{k}={v:.3f}" for k, v in st.items()))
    r = pd.DataFrame(draws)
    beat = int((r["avg_R"] >= real_stats["avg_R"]).sum())
    say(f"  -> real avg_R {real_stats['avg_R']:+.4f} vs random mean {r['avg_R'].mean():+.4f} "
        f"(range {r['avg_R'].min():+.4f}..{r['avg_R'].max():+.4f}); random beats real in {beat}/{N_DRAWS} draws")
    return real_trades, real_stats, r


def main():
    only = sys.argv[1:] or list(STRATEGIES)
    log = open(ml.OUT_DIR / "results_random_entry_all_bots.txt", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    rng = np.random.default_rng(0)
    segs = {"daily": daily_segments() if any(STRATEGIES[s]["data"] == "daily" for s in only) else [],
            "4h": fourh_segments() if "pcse" in only else []}
    saved = pickle.load(open(SAVE, "rb")) if Path(SAVE).exists() else {}
    for name in only:
        cfg = STRATEGIES[name]
        real_trades, _, _ = run_strategy(name, cfg, segs[cfg["data"]], say, rng)
        saved[name] = real_trades
        pickle.dump(saved, open(SAVE, "wb"))
    log.close()


if __name__ == "__main__":
    main()
