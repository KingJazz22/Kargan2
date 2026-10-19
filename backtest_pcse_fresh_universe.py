"""swing_pcse_exit on a fresh point-in-time S&P 500 universe, per the
pre-registered research_metalabel/SPEC_FRESH_UNIVERSE.md:
  H1 replication on names never used before (delisted included),
  H2 ATR%-based capital-efficiency rules in the 2%-risk shared ledger,
  H3 meta-labelers (logit / LightGBM / random forest / MLP) trained on the
     old 104-symbol trades and applied unchanged to the fresh names.
Writes research_metalabel/results_fresh.txt and _pcse_fresh_trades.pkl.
"""
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import backtest_metalabel_pcse as ml
from basket import US_SYMBOLS
from research_rotation.data_fetch import load_sp500_membership
from test_mtf_oos_symbols import OOS_US_SYMBOLS
from top100_symbols import TOP100_SYMBOLS

BARS_DIR = Path("research_rotation_data/alpaca_bars/all")
OLD_SET = set(TOP100_SYMBOLS + US_SYMBOLS + OOS_US_SYMBOLS)
SEGMENT_GAP, JUMP_BREAK = 10, np.log(4.0)
ATR_FLOOR = 0.02
SPLIT_DATE = pd.Timestamp("2021-01-01")
ALPHA_H3 = 0.05 / 4


def load_bars():
    parts = []
    for f in sorted(BARS_DIR.glob("*.parquet")):
        grp = f.stem.rsplit("_", 1)[0]
        if grp == "broad":
            continue
        d = pd.read_parquet(f)
        if d.empty:
            continue
        d["prio"] = 2 if grp.startswith("sp_former") else 1
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "t"])
    d["date"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    d.loc[d["symbol"].isin(OLD_SET), "prio"] = 0
    return d.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})


def membership_flags(d):
    m = load_sp500_membership()
    row_dates = np.array(sorted(m["date"].unique()), dtype="datetime64[ns]")
    member_rows = m.groupby("ticker")["date"].apply(lambda s: set(s.values.astype("datetime64[ns]")))
    p = np.searchsorted(row_dates, d["date"].values.astype("datetime64[ns]"), side="right") - 1
    row_of = np.where(p >= 0, row_dates[np.clip(p, 0, None)], np.datetime64("NaT"))
    out = np.zeros(len(d), dtype=bool)
    for sym, ix in d.groupby("symbol").indices.items():
        rows = member_rows.get(sym)
        if rows:
            out[ix] = [r in rows for r in row_of[ix]]
    return out


def dedupe(d):
    """Same date + close + volume under two tickers = one security; keep the highest-priority copy,
    member if any copy's ticker was a member that day."""
    d = d.copy()
    d["member"] = membership_flags(d)
    d["key"] = d["date"].astype("int64").astype(str) + "|" + d["Close"].round(4).astype(str) + "|" + d["Volume"].astype(str)
    d.loc[d["Volume"] <= 0, "key"] = d["symbol"] + "|" + d["key"]       # don't merge zero-volume prints
    any_member = d.groupby("key")["member"].transform("max")
    d = d.sort_values(["prio", "symbol"]).drop_duplicates("key", keep="first")
    d["member"] = any_member.loc[d.index]
    return d.sort_values(["symbol", "date"])


def split_segments(g, cal_pos):
    pos = cal_pos.reindex(g["date"]).to_numpy()
    lp = np.log(g["Close"].to_numpy())
    brk = np.r_[False, (np.diff(pos) > SEGMENT_GAP + 1) | (np.abs(np.diff(lp)) > JUMP_BREAK)]
    return np.cumsum(brk)


def build_all_trades():
    d = dedupe(load_bars())
    spy = d[d["symbol"] == "SPY"].set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
    cal = spy.index
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    spy_feat = ml.spy_features(spy)
    last_day = cal[-1]
    rows = []
    for sym, g in d.groupby("symbol"):
        if sym in ("SPY",) or not np.isfinite(g["Close"]).all():
            continue
        old = sym in OLD_SET
        g = g[g["date"].isin(cal)]
        for _, seg in g.groupby(split_segments(g, cal_pos)):
            if len(seg) <= ml.WARMUP + 60:
                continue
            df = seg.set_index("date")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            mask = None if old else seg.set_index("date")["member"]
            delisted = cal_pos[df.index[-1]] < len(cal) - 5
            for tr in ml.symbol_trades(sym, df, spy_feat, signal_mask=mask, liquidate_end=delisted):
                tr["set"] = "old" if old else "fresh"
                tr["still_member"] = bool(seg["member"].iloc[-1]) and df.index[-1] >= last_day - pd.Timedelta(days=7)
                rows.append(tr)
    t = pd.DataFrame(rows)
    t["y"] = (t["R"] > 0).astype(int)
    t = t.dropna(subset=ml.FEATURES).sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    return t


def tstat(R):
    return R.mean() / R.std(ddof=1) * np.sqrt(len(R)) if len(R) > 2 else np.nan


def ledger_stats(sub):
    s = ml.ledger(sub)
    return s["cagr"], s["sharpe"], s["max_dd"], s["n_trades"], s["total_return"]


def main():
    # --no-breaker: disable the ledger's latching 15% peak-DD circuit breaker (SPEC amendment 1)
    no_breaker = "--no-breaker" in sys.argv
    if no_breaker:
        ml.sw.CIRCUIT_BREAKER_DD_PCT = 1.0
    log = open(ml.OUT_DIR / ("results_fresh_nobreaker.txt" if no_breaker else "results_fresh.txt"), "w", encoding="utf-8")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")

    if Path("_pcse_fresh_trades.pkl").exists():
        with open("_pcse_fresh_trades.pkl", "rb") as f:
            t = pickle.load(f)
    else:
        t = build_all_trades()
        with open("_pcse_fresh_trades.pkl", "wb") as f:
            pickle.dump(t, f)
    old, fresh = t[t["set"] == "old"].reset_index(drop=True), t[t["set"] == "fresh"].reset_index(drop=True)

    # ---------------- H1
    say("=== H1 replication ===")
    for name, s in [("old-104", old), ("fresh", fresh)]:
        say(f"  {name:8s} n={len(s):5d} symbols={s['symbol'].nunique():4d} win={s['y'].mean():.3f} "
            f"meanR={s['R'].mean():+.4f} t={tstat(s['R']):.2f} exits={s['exit_reason'].value_counts().to_dict()}")
    early, late = fresh[fresh["entry_time"] < SPLIT_DATE], fresh[fresh["entry_time"] >= SPLIT_DATE]
    gone = fresh[~fresh["still_member"]]
    say(f"  fresh 2016-2020: n={len(early)} meanR={early['R'].mean():+.4f} t={tstat(early['R']):.2f}")
    say(f"  fresh 2021-2026: n={len(late)} meanR={late['R'].mean():+.4f} t={tstat(late['R']):.2f}")
    say(f"  fresh, names no longer in S&P (survivorship check): n={len(gone)} meanR={gone['R'].mean():+.4f} t={tstat(gone['R']):.2f}")
    h1 = tstat(fresh["R"]) >= 3 and early["R"].mean() > 0 and late["R"].mean() > 0 and gone["R"].mean() >= 0
    say(f"  H1 -> {'PASS' if h1 else 'FAIL'}")

    # ---------------- H2
    say("\n=== H2 capital efficiency (fresh set, 2% risk ledger) ===")
    sym_half = fresh["symbol"].map(lambda s: zlib.crc32(s.encode()) % 2)
    slices = {"full": np.ones(len(fresh), bool),
              "2016-2020": (fresh["entry_time"] < SPLIT_DATE).to_numpy(),
              "2021-2026": (fresh["entry_time"] >= SPLIT_DATE).to_numpy(),
              "symA": (sym_half == 0).to_numpy(), "symB": (sym_half == 1).to_numpy()}

    def ordered(sub, rule):
        if rule == "rank":
            return sub.assign(_k=-sub["atr_pct"]).sort_values(["entry_time", "_k"]).drop(columns="_k")
        if rule == "floor":
            sub = sub[sub["atr_pct"] >= ATR_FLOOR]
        return sub.sort_values(["entry_time", "symbol"])

    res = {}
    for rule in ["baseline", "rank", "floor"]:
        for sl_name, m in slices.items():
            res[(rule, sl_name)] = ledger_stats(ordered(fresh[m], rule))
            c, sh, dd, n, tr = res[(rule, sl_name)]
            say(f"  {rule:8s} {sl_name:9s} taken={n:4d} return={tr:+7.1f}% CAGR={c:5.1f}% sharpe={sh:.2f} maxDD={dd:.1f}%")
    for rule in ["rank", "floor"]:
        ok = all(res[(rule, s)][0] > res[("baseline", s)][0] and res[(rule, s)][1] > res[("baseline", s)][1] for s in slices)
        ok = ok and res[(rule, "full")][2] >= res[("baseline", "full")][2] - 3.0
        say(f"  H2 {rule} -> {'PASS' if ok else 'FAIL'}")

    # ---------------- H3
    say("\n=== H3 meta-labeler transfer (fit on old-104 trades, applied to fresh trades) ===")
    models = ml.make_models()
    models["rf"] = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features="sqrt",
                                          class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    models["mlp"] = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                                                  early_stopping=True, max_iter=500, random_state=0))
    Xo, yo = old[ml.FEATURES].to_numpy(), old["y"].to_numpy()
    Xf, Rf, yf = fresh[ml.FEATURES].to_numpy(), fresh["R"].to_numpy(), fresh["y"].to_numpy()
    base_cagr = res[("baseline", "full")][0]
    rng = np.random.default_rng(0)
    for name, model in models.items():
        model.fit(Xo, yo)
        thr = np.quantile(model.predict_proba(Xo)[:, 1], ml.FILTER_Q)
        p = model.predict_proba(Xf)[:, 1]
        w = (p >= thr).astype(float)
        obs, pval, ci = ml.block_bootstrap_p(fresh, w, rng)
        e = (fresh["entry_time"] < SPLIT_DATE).to_numpy()
        d_e = ml.weighted_r(Rf[e], w[e]) - Rf[e].mean()
        d_l = ml.weighted_r(Rf[~e], w[~e]) - Rf[~e].mean()
        kept = w > 0
        c, sh, dd, n, tr = ledger_stats(fresh[kept])
        ok = obs > 0 and pval < ALPHA_H3 and d_e > 0 and d_l > 0 and c >= base_cagr
        say(f"  {name:5s} AUC={roc_auc_score(yf, p):.3f} kept={kept.sum()}/{len(fresh)} "
            f"losers skipped={((~kept) & (yf == 0)).sum()}/{(yf == 0).sum()} delta={obs:+.4f} "
            f"95%CI[{ci[0]:+.4f},{ci[1]:+.4f}] p={pval:.4f} halves {d_e:+.4f}/{d_l:+.4f} "
            f"ledger CAGR={c:.1f}% (base {base_cagr:.1f}%) sharpe={sh:.2f} maxDD={dd:.1f}% -> {'PASS' if ok else 'FAIL'}")
    log.close()


if __name__ == "__main__":
    main()
