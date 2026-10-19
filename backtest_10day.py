"""swing_pcse entries with exits built for a max-10-day hold, per the
pre-registered research_metalabel/SPEC_10DAY.md.

  python backtest_10day.py            # stage 1: S&P development grid -> selects one config
  python backtest_10day.py --holdout  # stage 2: selected config vs baseline on unused non-S&P stocks

Writes research_metalabel/results_10day_{dev,holdout}.txt and _10day_selected.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_metalabel_pcse as ml
import backtest_pcse_fresh_universe as fu
import strategy_swing_breakout as strat_swing
import sweep_risk_levels as sw

MAX_DAYS = 10
TP_GRID, SL_GRID = [1.0, 1.5, 2.0, 3.0], [1.5, 2.0, 3.0, 5.0]
BASELINE = {"name": "baseline sl14/tp3/no-limit", "sl": 14.0, "tp": 3.0, "max_days": None}
SELECTED_FILE = Path("_10day_selected.json")
BROAD_MIN_PRICE, BROAD_MIN_DV = 5.0, 20e6

# live breaker rule (circuit_breaker.py): re-arm after 30 days, half risk until the old peak is back
sw.CIRCUIT_BREAKER_COOLDOWN_DAYS, sw.CIRCUIT_BREAKER_REARM_RISK_MULT = 30, 0.5


def configs():
    out = [BASELINE]
    for tp in TP_GRID:
        for sl in SL_GRID:
            out.append({"name": f"sl{sl:g}/tp{tp:g}/{MAX_DAYS}d", "sl": sl, "tp": tp, "max_days": MAX_DAYS})
    return out


def prepared_segments(d, mask_fn):
    """[(symbol, prepared df, liquidate_end)] -- signals already masked by mask_fn(symbol, seg)."""
    spy = d[d["symbol"] == "SPY"].set_index("date")
    cal = spy.index
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    out = []
    for sym, g in d.groupby("symbol"):
        if sym == "SPY":
            continue
        g = g[g["date"].isin(cal)]
        if len(g) <= ml.WARMUP + 60 or not np.isfinite(g["Close"]).all():
            continue
        for _, seg in g.groupby(fu.split_segments(g, cal_pos)):
            if len(seg) <= ml.WARMUP + 60:
                continue
            df = seg.set_index("date")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            p = strat_swing.prepare(df)
            m = mask_fn(sym, seg.set_index("date"))
            if m is not None:
                p["long_entry"] = p["long_entry"] & m.reindex(p.index).fillna(False).astype(bool)
            out.append((sym, p, cal_pos[df.index[-1]] < len(cal) - 5))
    return out


def simulate(sym, p, liquidate_end, sl_mult, tp_mult, max_days):
    o, h, l, c = (p[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    sig, atr_ = p["long_entry"].to_numpy(), p["atr"].to_numpy()
    idx = p.index
    out, i, n = [], ml.WARMUP, len(p)
    while i < n - 1:
        if not sig[i] or not np.isfinite(atr_[i]) or atr_[i] <= 0:
            i += 1
            continue
        e = i + 1
        entry = o[e] * (1 + ml.COST)
        sl, tp = entry - sl_mult * atr_[i], entry + tp_mult * atr_[i]
        limit = idx[e] + pd.Timedelta(days=max_days) if max_days else None
        exit_px = reason = None
        j = e
        while j < n:
            if l[j] <= sl:
                exit_px, reason = (o[j] if o[j] <= sl else sl), "stop_loss"
            elif h[j] >= tp:
                exit_px, reason = (o[j] if o[j] >= tp else tp), "take_profit"
            elif limit is not None and idx[j] >= limit:
                exit_px, reason = c[j], "time_stop"
            if reason:
                break
            j += 1
        if reason is None:
            if not liquidate_end:
                break
            j, exit_px, reason = n - 1, c[n - 1], "delisted"
        exit_px *= (1 - ml.COST)
        out.append({"symbol": sym, "entry_time": idx[e], "exit_time": idx[j], "entry_price": entry,
                    "exit_price": exit_px, "stop_distance": sl_mult * atr_[i], "exit_reason": reason,
                    "R": (exit_px - entry) / (sl_mult * atr_[i])})
        i = j
    return out


def evaluate(segs, cfg):
    rows = []
    for sym, p, liq_end in segs:
        rows += simulate(sym, p, liq_end, cfg["sl"], cfg["tp"], cfg["max_days"])
    t = pd.DataFrame(rows).sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    R = t["R"].to_numpy()
    hold = (t["exit_time"] - t["entry_time"]).dt.days.clip(lower=1)
    s = ml.ledger(t)
    return {"config": cfg["name"], "trades": len(t), "win%": (R > 0).mean() * 100, "avg_R": R.mean(),
            "t": R.mean() / R.std(ddof=1) * np.sqrt(len(R)), "avg_hold_d": hold.mean(),
            "time_stop%": (t["exit_reason"] == "time_stop").mean() * 100,
            "avg_ret%": (t["exit_price"] / t["entry_price"] - 1).mean() * 100,
            "taken": s["n_trades"], "CAGR%": s["cagr"], "sharpe": s["sharpe"], "maxDD%": s["max_dd"]}


def sp500_segments():
    d = fu.dedupe(fu.load_bars())
    return prepared_segments(d, lambda sym, seg: None if sym in fu.OLD_SET else seg["member"])


def broad_segments():
    parts = [pd.read_parquet(f) for f in sorted(fu.BARS_DIR.glob("broad_*.parquet"))]
    b = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "t"])
    b["date"] = pd.to_datetime(b["t"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    b = b.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    sp = fu.load_bars()
    key = lambda x: x["date"].astype("int64").astype(str) + "|" + x["Close"].round(4).astype(str) + "|" + x["Volume"].astype(str)
    b = b[~key(b).isin(set(key(sp[sp["Volume"] > 0])))]          # drop duplicate listings of S&P securities
    spy = sp[sp["symbol"] == "SPY"]
    d = pd.concat([b, spy], ignore_index=True).sort_values(["symbol", "date"])

    def liquid(sym, seg):
        dv = (seg["Close"] * seg["Volume"]).rolling(63, min_periods=40).median()
        return (seg["Close"] >= BROAD_MIN_PRICE) & (dv >= BROAD_MIN_DV)
    return prepared_segments(d, liquid)


def main():
    holdout = "--holdout" in sys.argv
    log = open(ml.OUT_DIR / f"results_10day_{'holdout' if holdout else 'dev'}.txt", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    if not holdout:
        segs = sp500_segments()
        res = pd.DataFrame([evaluate(segs, c) for c in configs()])
        say("STAGE 1 -- development, S&P point-in-time, 2% risk ledger with live breaker rule")
        say(res.round(3).to_string(index=False))
        ok = res[(res["config"] != BASELINE["name"]) & (res["t"] >= 3) & (res["maxDD%"] >= -20)]
        base_cagr = res.loc[res["config"] == BASELINE["name"], "CAGR%"].iloc[0]
        if ok.empty:
            say("\nno 10-day config meets t>=3 and maxDD>=-20% -> nothing goes to the holdout")
            return
        best = ok.sort_values("CAGR%", ascending=False).iloc[0]
        go = best["CAGR%"] > base_cagr
        say(f"\nselected: {best['config']} (CAGR {best['CAGR%']:.1f}% vs baseline {base_cagr:.1f}%) -> "
            f"{'goes to holdout' if go else 'does NOT beat baseline; holdout not run'}")
        if go:
            cfg = next(c for c in configs() if c["name"] == best["config"])
            SELECTED_FILE.write_text(json.dumps(cfg))
    else:
        if not SELECTED_FILE.exists():
            sys.exit("no selected config -- run stage 1 first")
        cfg = json.loads(SELECTED_FILE.read_text())
        segs = broad_segments()
        say(f"STAGE 2 -- holdout, non-S&P liquid stocks (price >= ${BROAD_MIN_PRICE:g}, 63d median $vol >= "
            f"${BROAD_MIN_DV / 1e6:g}M), {len({s for s, _, _ in segs})} symbols")
        res = pd.DataFrame([evaluate(segs, BASELINE), evaluate(segs, cfg)])
        say(res.round(3).to_string(index=False))
        b, s = res.iloc[0], res.iloc[1]
        ok = s["CAGR%"] > b["CAGR%"] and s["sharpe"] > b["sharpe"] and s["t"] >= 3 and s["maxDD%"] >= -20
        say(f"\n{cfg['name']} vs baseline -> {'PASS' if ok else 'FAIL'}")
    log.close()


if __name__ == "__main__":
    main()
