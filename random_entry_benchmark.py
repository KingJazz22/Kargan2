"""Does the structure-confirmed breakout ENTRY add anything over random entries
with the same exit (14xATR stop, 3xATR target, no time limit)?

Same S&P point-in-time universe, engine and costs as backtest_10day.py's
baseline. Each random draw puts the same number of signals as the real
strategy into each symbol segment, on uniformly random days among the days
the real signal was allowed to fire (index member, past warmup). Also prints
the driftless random-walk benchmark for these barriers: P(target first) =
SL/(SL+TP), E[R] = 0.

Writes research_metalabel/results_random_entry.txt.
"""
import numpy as np
import pandas as pd

import backtest_10day as tday
import backtest_metalabel_pcse as ml
import backtest_pcse_fresh_universe as fu

N_DRAWS = 20
SL, TP = 14.0, 3.0


def main():
    log = open(ml.OUT_DIR / "results_random_entry.txt", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    d = fu.dedupe(fu.load_bars())
    segs = tday.prepared_segments(d, lambda sym, seg: None if sym in fu.OLD_SET else seg["member"])
    # days a signal was allowed: past warmup and (for fresh names) S&P member -- recover from an unmasked copy
    allowed = {}
    for k, (sym, p, _) in enumerate(segs):
        ok = np.zeros(len(p), dtype=bool)
        ok[ml.WARMUP:len(p) - 1] = True
        if sym not in fu.OLD_SET:
            mem = d[d["symbol"] == sym].set_index("date")["member"].reindex(p.index).fillna(False).to_numpy(bool)
            ok &= mem
        ok &= np.isfinite(p["atr"].to_numpy()) & (p["atr"].to_numpy() > 0)
        allowed[k] = np.flatnonzero(ok)

    def run(signal_fn):
        rows = []
        for k, (sym, p, liq_end) in enumerate(segs):
            q = p.copy()
            q["long_entry"] = signal_fn(k, q)
            rows += tday.simulate(sym, q, liq_end, SL, TP, None)
        t = pd.DataFrame(rows)
        R = t["R"].to_numpy()
        hold = (t["exit_time"] - t["entry_time"]).dt.days.clip(lower=1)
        return {"trades": len(t), "win%": (R > 0).mean() * 100, "avg_R": R.mean(),
                "R_per_100d": R.sum() / hold.sum() * 100, "avg_hold_d": hold.mean()}

    say(f"Driftless random walk, continuous prices: P(target before stop) = {SL}/({SL}+{TP}) = "
        f"{SL / (SL + TP):.1%}, E[R] = 0\n")
    real = run(lambda k, q: q["long_entry"].to_numpy())
    say(f"real breakout entries: {real}")

    rng = np.random.default_rng(0)
    draws = []
    for b in range(N_DRAWS):
        def rand_sig(k, q):
            n_sig = int(q["long_entry"].sum())
            out = np.zeros(len(q), dtype=bool)
            if n_sig and len(allowed[k]):
                out[rng.choice(allowed[k], size=min(n_sig, len(allowed[k])), replace=False)] = True
            return out
        draws.append(run(rand_sig))
        say(f"  random draw {b + 1:2d}: " + " ".join(f"{k}={v:.3f}" for k, v in draws[-1].items()))
    r = pd.DataFrame(draws)
    say(f"\nrandom entries, {N_DRAWS} draws: avg_R mean {r['avg_R'].mean():+.4f} (min {r['avg_R'].min():+.4f}, "
        f"max {r['avg_R'].max():+.4f}); win% {r['win%'].mean():.1f}; R/100d {r['R_per_100d'].mean():.3f}")
    say(f"real minus random: avg_R {real['avg_R'] - r['avg_R'].mean():+.4f}; draws beating real: "
        f"{(r['avg_R'] >= real['avg_R']).sum()}/{N_DRAWS}")
    log.close()


if __name__ == "__main__":
    main()
