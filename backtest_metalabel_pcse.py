"""Meta-labeling test on swing_pcse_exit (sl=14xATR, TP=3xATR fixed), per the
pre-registered research_metalabel/SPEC.md. Primary signal unchanged; a
secondary classifier predicts whether each entry wins, validated with
combinatorial purged CV (N=6, k=2, purge + 10-day embargo). Writes
research_metalabel/results.txt and _metalabel_trades.pkl.
"""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import strategy_swing_breakout as strat_swing
import sweep_risk_levels as sw
from basket import US_SYMBOLS
from metalabel_features import FEATURES, feature_frame, spy_features  # noqa: F401 (re-exported)
from split_adjust import load_daily_cache_adjusted
from test_mtf_oos_symbols import OOS_US_SYMBOLS
from top100_symbols import TOP100_SYMBOLS

SL_ATR, TP_ATR, COST, WARMUP = 14.0, 3.0, 0.0005, 70
N_GROUPS, K_TEST, EMBARGO_DAYS = 6, 2, 10
FILTER_Q, N_BOOT, ALPHA = 0.20, 5000, 0.05 / 4
RISK_PCT = 0.02
OUT_DIR = Path("research_metalabel")


def make_models():
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        "lgbm": LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=30,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, is_unbalance=True,
                               random_state=0, verbose=-1),
    }


def symbol_trades(symbol, df, spy_feat, signal_mask=None, liquidate_end=False):
    """signal_mask: optional bool Series on df's index (e.g. point-in-time index
    membership). liquidate_end: exit a still-open trade at the last close
    (delisting / segment end) instead of dropping it."""
    p = strat_swing.prepare(df)
    if signal_mask is not None:
        p["long_entry"] = p["long_entry"] & signal_mask.reindex(p.index).fillna(False).astype(bool)
    feat = feature_frame(p, spy_feat)

    o, h, l = p["Open"].to_numpy(), p["High"].to_numpy(), p["Low"].to_numpy()
    sig, atr_ = p["long_entry"].to_numpy(), p["atr"].to_numpy()
    idx = p.index
    out, i, n = [], WARMUP, len(p)
    while i < n - 1:
        if not sig[i] or not np.isfinite(atr_[i]) or atr_[i] <= 0:
            i += 1
            continue
        e = i + 1
        entry = o[e] * (1 + COST)
        sl, tp = entry - SL_ATR * atr_[i], entry + TP_ATR * atr_[i]
        exit_px = reason = None
        j = e
        while j < n:
            if l[j] <= sl:
                exit_px, reason = (o[j] if o[j] <= sl else sl), "stop_loss"
            elif h[j] >= tp:
                exit_px, reason = (o[j] if o[j] >= tp else tp), "take_profit"
            if reason:
                break
            j += 1
        if reason is None:
            if not liquidate_end:
                break  # still open at the end of data
            j = n - 1
            exit_px, reason = p["Close"].iloc[j], "delisted"
        exit_px *= (1 - COST)
        row = {"symbol": symbol, "signal_date": idx[i], "entry_time": idx[e], "exit_time": idx[j],
               "entry_price": entry, "exit_price": exit_px, "stop_distance": SL_ATR * atr_[i],
               "exit_reason": reason}
        row["R"] = (exit_px - entry) / row["stop_distance"]
        row.update(feat.iloc[i].to_dict())
        out.append(row)
        i = j  # next signal can fire on the exit bar's close, as in the engine
    return out


def build_trades():
    cache = load_daily_cache_adjusted()
    spy_feat = spy_features(cache["SPY"])
    symbols = list(dict.fromkeys(TOP100_SYMBOLS + US_SYMBOLS + OOS_US_SYMBOLS))
    rows = []
    for s in symbols:
        df = cache.get(s)
        if df is None or len(df) <= WARMUP + 60:
            continue
        rows += symbol_trades(s, df, spy_feat)
    t = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    t["y"] = (t["R"] > 0).astype(int)
    return t.dropna(subset=FEATURES).reset_index(drop=True)


def _overlaps_any(entry, exit_, lo, hi):
    """Bool per (entry, exit_) row: does it overlap any [lo_i, hi_i] interval? Merges the intervals first."""
    order = np.argsort(lo)
    merged = []
    for a, b in zip(lo[order], hi[order]):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    m_lo, m_hi = np.array([m[0] for m in merged]), np.array([m[1] for m in merged])
    k = np.searchsorted(m_lo, exit_, side="right") - 1   # last merged interval starting at/before exit
    return (k >= 0) & (m_hi[np.clip(k, 0, None)] >= entry)


def cpcv_splits(t, exact=False):
    """Yields (train_idx, test_groups, group_of) for each of C(N,k) splits, purged + embargoed.
    exact=False purges against each test group's whole [first entry, last exit] span (conservative;
    what SPEC.md's run used). exact=True purges only training trades overlapping an actual test
    trade's holding period, and embargoes after the group's last entry -- needed once multi-year
    holds would otherwise make the span swallow almost the whole sample."""
    if exact:
        groups = np.array_split(np.arange(len(t)), N_GROUPS)
        group_of = np.empty(len(t), dtype=int)
        for g, ix in enumerate(groups):
            group_of[ix] = g
        days = pd.DatetimeIndex(sorted(t["entry_time"].unique())).values
        entry, exit_ = t["entry_time"].values, t["exit_time"].values
        for test in itertools.combinations(range(N_GROUPS), K_TEST):
            te = np.isin(group_of, test)
            keep = ~te & ~_overlaps_any(entry, exit_, entry[te], exit_[te])
            for g in test:
                last = entry[groups[g]].max()
                emb_end = days[min(np.searchsorted(days, last) + EMBARGO_DAYS, len(days) - 1)]
                keep &= ~((entry > last) & (entry <= emb_end))
            yield np.where(keep)[0], test, group_of
        return
    groups = np.array_split(np.arange(len(t)), N_GROUPS)
    group_of = np.empty(len(t), dtype=int)
    for g, ix in enumerate(groups):
        group_of[ix] = g
    days = pd.DatetimeIndex(sorted(t["entry_time"].unique()))
    entry, exit_ = t["entry_time"], t["exit_time"]
    for test in itertools.combinations(range(N_GROUPS), K_TEST):
        keep = ~np.isin(group_of, test)
        for g in test:
            ix = groups[g]
            lo, hi = entry.iloc[ix].min(), exit_.iloc[ix].max()
            emb_end = days[min(days.searchsorted(hi) + EMBARGO_DAYS, len(days) - 1)]
            overlap = ((entry <= hi) & (exit_ >= lo)).to_numpy()
            embargo = ((entry > hi) & (entry <= emb_end)).to_numpy()
            keep &= ~overlap & ~embargo
        yield np.where(keep)[0], test, group_of


def assign_paths():
    """Each group is tested in C(N-1,k-1) splits; the m-th of them feeds path m."""
    splits = list(itertools.combinations(range(N_GROUPS), K_TEST))
    counter = {g: 0 for g in range(N_GROUPS)}
    path_of = {}
    for s_i, test in enumerate(splits):
        for g in test:
            path_of[(s_i, g)] = counter[g]
            counter[g] += 1
    return path_of, counter[0]


def weighted_r(R, w):
    return np.sum(w * R) / np.sum(w) if np.sum(w) > 0 else np.nan


def policy_weights(prob, train_prob, policy):
    if policy == "filter":
        return (prob >= np.quantile(train_prob, FILTER_Q)).astype(float)
    return 2.0 * np.searchsorted(np.sort(train_prob), prob, side="right") / len(train_prob)


def block_bootstrap_p(t, w, rng):
    """One-sided p for delta = weighted_r(policy) - mean(R), resampling calendar months."""
    month = t["entry_time"].dt.tz_localize(None).dt.to_period("M").to_numpy()
    blocks = [np.where(month == m)[0] for m in np.unique(month)]
    R = t["R"].to_numpy()
    obs = weighted_r(R, w) - R.mean()
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ix = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
        deltas[b] = weighted_r(R[ix], w[ix]) - R[ix].mean()
    # p = share of bootstrap deltas, re-centred under H0 (delta=0), at or above the observed delta
    return obs, np.mean(deltas - deltas.mean() >= obs), np.percentile(deltas, [2.5, 97.5])


def psr(R, benchmark=0.0):
    sr = R.mean() / R.std(ddof=1)
    g3, g4 = stats.skew(R), stats.kurtosis(R, fisher=False)
    return stats.norm.cdf((sr - benchmark) * np.sqrt(len(R) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))


def ledger(trades_df):
    recs = [{"strategy_key": "swing_pcse_exit", "symbol": r.symbol,
             "entry_time": sw.to_utc(r.entry_time), "exit_time": sw.to_utc(r.exit_time),
             "entry_price": r.entry_price, "exit_price": r.exit_price,
             "pnl_pct": r.exit_price / r.entry_price - 1, "stop_distance": r.stop_distance}
            for r in trades_df.itertuples()]
    closed, eq, taken, bc, bd = sw.replay_shared_ledger(recs, RISK_PCT)
    return sw.summarize_replay(RISK_PCT, closed, eq, taken, bc, bd)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    log = open(OUT_DIR / "results.txt", "w", encoding="utf-8")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")

    t = build_trades()
    R, y = t["R"].to_numpy(), t["y"].to_numpy()
    X = t[FEATURES].to_numpy()
    say(f"trades={len(t)}  symbols={t['symbol'].nunique()}  {t['entry_time'].min().date()} -> {t['entry_time'].max().date()}")
    say(f"baseline: win={y.mean():.3f}  mean R={R.mean():+.4f}  t={R.mean() / R.std(ddof=1) * np.sqrt(len(R)):.2f}  "
        f"PSR(>0)={psr(R):.3f}  exits={t['exit_reason'].value_counts().to_dict()}")

    path_of, n_paths = assign_paths()
    is_mask = t["symbol"].isin(US_SYMBOLS).to_numpy()
    rng = np.random.default_rng(0)
    summary = []

    for mname in ["logit", "lgbm"]:
        probs = np.full((n_paths, len(t)), np.nan)        # OOS prob per path
        trainq = np.full((n_paths, len(t)), np.nan)       # filter threshold used for that prediction
        size_w = np.full((n_paths, len(t)), np.nan)
        aucs, importances = [], []
        for s_i, (train_idx, test, group_of) in enumerate(cpcv_splits(t)):
            model = make_models()[mname]
            model.fit(X[train_idx], y[train_idx])
            train_p = model.predict_proba(X[train_idx])[:, 1]
            if mname == "lgbm":
                importances.append(model.booster_.feature_importance("gain"))
            for g in test:
                ix = np.where(group_of == g)[0]
                p = model.predict_proba(X[ix])[:, 1]
                m = path_of[(s_i, g)]
                probs[m, ix] = p
                trainq[m, ix] = np.quantile(train_p, FILTER_Q)
                size_w[m, ix] = policy_weights(p, train_p, "size")
                if len(np.unique(y[ix])) == 2:
                    aucs.append(roc_auc_score(y[ix], p))
        say(f"\n=== {mname} ===  OOS AUC per test group: mean {np.mean(aucs):.3f}  "
            f"min {np.min(aucs):.3f}  max {np.max(aucs):.3f}")
        if importances:
            imp = pd.Series(np.mean(importances, axis=0), index=FEATURES).sort_values(ascending=False)
            say("  top features (gain):", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(6).items()))

        pooled_p = probs.mean(axis=0)
        for policy in ["filter", "size"]:
            path_w = (probs >= trainq).astype(float) if policy == "filter" else size_w
            path_d = [weighted_r(R, path_w[m]) - R.mean() for m in range(n_paths)]
            pooled_w = (pooled_p >= trainq.mean(axis=0)).astype(float) if policy == "filter" else size_w.mean(axis=0)
            obs, pval, ci = block_bootstrap_p(t, pooled_w, rng)
            d_is = weighted_r(R[is_mask], pooled_w[is_mask]) - R[is_mask].mean()
            d_oos = weighted_r(R[~is_mask], pooled_w[~is_mask]) - R[~is_mask].mean()
            kept = pooled_w > 0
            say(f"  [{policy}] delta R/risk per path: " + " ".join(f"{d:+.4f}" for d in path_d))
            say(f"  [{policy}] pooled: policy R/risk={weighted_r(R, pooled_w):+.4f} vs base {R.mean():+.4f}  "
                f"delta={obs:+.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  p={pval:.4f}  "
                f"IS-sym delta={d_is:+.4f}  OOS-sym delta={d_oos:+.4f}")
            if policy == "filter":
                say(f"  [filter] kept {kept.sum()}/{len(t)}; win rate kept={y[kept].mean():.3f} "
                    f"skipped={y[~kept].mean():.3f}; losers skipped {((~kept) & (y == 0)).sum()}/{(y == 0).sum()}")
            ok = [all(d > 0 for d in path_d), pval < ALPHA, d_is > 0 and d_oos > 0]
            ledger_ok = None
            if policy == "filter":
                base_s, filt_s = ledger(t), ledger(t[kept])
                say(f"  [filter] ledger 2% risk: base return={base_s['total_return']:+.1f}% maxDD={base_s['max_dd']:.1f}% "
                    f"sharpe={base_s['sharpe']:.2f} | filtered return={filt_s['total_return']:+.1f}% "
                    f"maxDD={filt_s['max_dd']:.1f}% sharpe={filt_s['sharpe']:.2f}")
                ledger_ok = filt_s["total_return"] >= base_s["total_return"] and filt_s["max_dd"] >= base_s["max_dd"]
                ok.append(ledger_ok)
            verdict = "PASS" if all(ok) else "FAIL"
            say(f"  [{policy}] criteria (all paths>0, p<{ALPHA:.4f}, both halves>0{', ledger' if policy == 'filter' else ''}): "
                f"{ok} -> {verdict}")
            summary.append((mname, policy, obs, pval, verdict))
        t[f"p_{mname}"] = pooled_p

    say("\n=== SUMMARY ===")
    for row in summary:
        say(f"  {row[0]:6s} {row[1]:7s} delta={row[2]:+.4f}  p={row[3]:.4f}  {row[4]}")
    with open("_metalabel_trades.pkl", "wb") as f:
        pickle.dump(t, f)
    log.close()


if __name__ == "__main__":
    sys.exit(main())
