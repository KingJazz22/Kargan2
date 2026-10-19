"""Pooled vs grouped random-forest meta-labelers on all swing_pcse_exit trades
(old 104 + fresh 580 S&P names), per research_metalabel/SPEC_RF_GROUPED.md
(yearly walk-forward, training only on trades already closed -- amendment 2).
Design comparison only -- the trades were already used, so a winner is a
shadow-log candidate, not a validated filter. Writes
research_metalabel/results_rf_grouped.txt.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import backtest_metalabel_pcse as ml
import backtest_pcse_fresh_universe as fu

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
RF = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features="sqrt",
                            class_weight="balanced_subsample", random_state=0, n_jobs=-1)
MIN_GROUP_LOSERS, STOCK_MIN_TRADES, STOCK_MIN_LOSERS, LOSS_PRIOR = 30, 40, 5, 20
TEST_YEARS = list(range(2019, 2027))
DESIGNS = ["A_pooled", "B_pooled+groups", "C_per_liquidity", "D_per_sector", "E_per_regime", "F_per_stock", "G_blend"]


def group_features(t):
    bars = fu.load_bars()
    close = bars.pivot_table(index="date", columns="symbol", values="Close", aggfunc="last").sort_index()
    vol = bars.pivot_table(index="date", columns="symbol", values="Volume", aggfunc="last").reindex_like(close)
    dv = (close * vol).rolling(63, min_periods=40).median()
    liq_pct = dv.rank(axis=1, pct=True)
    rets = close.pct_change(fill_method=None)
    etf = rets[SECTOR_ETFS]
    pos = close.index.get_indexer(pd.DatetimeIndex(t["signal_date"]))
    out_liq, out_sec = np.full(len(t), np.nan), []
    for k, (sym, p) in enumerate(zip(t["symbol"], pos)):
        out_liq[k] = liq_pct.iat[p, liq_pct.columns.get_loc(sym)] if p >= 0 and sym in liq_pct.columns else np.nan
        if p < 252 or sym not in rets.columns:
            out_sec.append("unknown")
            continue
        r = rets[sym].iloc[p - 251:p + 1]
        e = etf.iloc[p - 251:p + 1]
        ok = r.notna()
        corr = e[ok].corrwith(r[ok]) if ok.sum() >= 120 else pd.Series(dtype=float)
        out_sec.append(corr.idxmax() if corr.notna().any() else "unknown")
    t = t.copy()
    t["liq_pct"] = out_liq
    t["liq_grp"] = pd.cut(t["liq_pct"], [-np.inf, 1 / 3, 2 / 3, np.inf], labels=["low", "mid", "high"]).astype(str)
    t["sector"] = out_sec
    t["regime"] = np.where(t["spy_above_sma200"] > 0.5, "bull", "bear")
    return t


def fit_predict(X, y, tr, te):
    m = clone(RF).fit(X[tr], y[tr])
    return m.predict_proba(X[tr])[:, 1], m.predict_proba(X[te])[:, 1]


def per_group(X, y, tr, te, groups, base_in, base_out, min_losers=MIN_GROUP_LOSERS, min_trades=0):
    """Separate model per group; groups too thin to train fall back to the pooled model."""
    p_in, p_out = base_in.copy(), base_out.copy()
    n_models = 0
    for g in np.unique(groups[te]):
        gtr, gte = tr[groups[tr] == g], te[groups[te] == g]
        if (y[gtr] == 0).sum() < min_losers or len(gtr) < min_trades or len(np.unique(y[gtr])) < 2:
            continue
        pi, po = fit_predict(X, y, gtr, gte)
        p_in[np.searchsorted(tr, gtr)] = pi
        p_out[np.searchsorted(te, gte)] = po
        n_models += 1
    return p_in, p_out, n_models


def main():
    log = open(ml.OUT_DIR / "results_rf_grouped.txt", "w", encoding="utf-8")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")

    t = pd.read_pickle("_pcse_fresh_trades.pkl").sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    t = group_features(t)
    say(f"trades={len(t)} losers={(t['y'] == 0).sum()} symbols={t['symbol'].nunique()}")
    say("sector counts:", t["sector"].value_counts().to_dict())
    say("loss rate by liquidity tercile:", t.groupby("liq_grp")["y"].apply(lambda s: round(1 - s.mean(), 3)).to_dict())
    say("loss rate by sector:", t.groupby("sector")["y"].apply(lambda s: round(1 - s.mean(), 3)).to_dict())
    say("loss rate by regime:", t.groupby("regime")["y"].apply(lambda s: round(1 - s.mean(), 3)).to_dict())

    y, R = t["y"].to_numpy(), t["R"].to_numpy()
    Xb = t[ml.FEATURES].to_numpy()
    sector_oh = pd.get_dummies(t["sector"]).to_numpy(float)
    liq, sec, reg, stk = (t[c].to_numpy() for c in ["liq_grp", "sector", "regime", "symbol"])

    probs = {d: np.full(len(t), np.nan) for d in DESIGNS}
    thr = {d: np.full(len(t), np.nan) for d in DESIGNS}
    aucs = {d: [] for d in DESIGNS}
    stock_models = []
    entry_year = t["entry_time"].dt.year.to_numpy()

    # SPEC amendment 2: yearly walk-forward, training only on trades already closed (label known)
    for s_i, year in enumerate(TEST_YEARS):
        tr = np.where(t["exit_time"] < pd.Timestamp(f"{year}-01-01"))[0]
        te = np.where(entry_year == year)[0]
        out_in, out_te = {}, {}
        out_in["A_pooled"], out_te["A_pooled"] = fit_predict(Xb, y, tr, te)

        # B: group features + shrunk per-stock loss rate from training trades only
        g_rate = 1 - y[tr].mean()
        st = pd.DataFrame({"s": stk[tr], "loss": 1 - y[tr]}).groupby("s")["loss"].agg(["sum", "count"])
        shrunk = (st["sum"] + LOSS_PRIOR * g_rate) / (st["count"] + LOSS_PRIOR)
        stock_rate = pd.Series(stk).map(shrunk).fillna(g_rate).to_numpy()
        XB = np.column_stack([Xb, t["liq_pct"].fillna(0.5).to_numpy(), sector_oh, stock_rate])
        out_in["B_pooled+groups"], out_te["B_pooled+groups"] = fit_predict(XB, y, tr, te)

        a_in, a_te = out_in["A_pooled"], out_te["A_pooled"]
        out_in["C_per_liquidity"], out_te["C_per_liquidity"], _ = per_group(Xb, y, tr, te, liq, a_in, a_te)
        out_in["D_per_sector"], out_te["D_per_sector"], _ = per_group(Xb, y, tr, te, sec, a_in, a_te)
        out_in["E_per_regime"], out_te["E_per_regime"], _ = per_group(Xb, y, tr, te, reg, a_in, a_te)
        out_in["F_per_stock"], out_te["F_per_stock"], n_stock = per_group(
            Xb, y, tr, te, stk, a_in, a_te, min_losers=STOCK_MIN_LOSERS, min_trades=STOCK_MIN_TRADES)
        stock_models.append(n_stock)
        blend = ["A_pooled", "C_per_liquidity", "D_per_sector", "E_per_regime"]
        out_in["G_blend"] = np.mean([out_in[d] for d in blend], axis=0)
        out_te["G_blend"] = np.mean([out_te[d] for d in blend], axis=0)

        for d in DESIGNS:
            aucs[d].append(roc_auc_score(y[te], out_te[d]))
            probs[d][te] = out_te[d]
            thr[d][te] = np.quantile(out_in[d], ml.FILTER_Q)
        say(f"  {year} (train {len(tr)}, test {len(te)}): " + " ".join(f"{d.split('_')[0]}={aucs[d][-1]:.3f}" for d in DESIGNS)
            + f" | per-stock models trained: {n_stock}")

    say(f"\nstocks qualifying for their own model (>= {STOCK_MIN_TRADES} trades and >= {STOCK_MIN_LOSERS} losers "
        f"in a training window): {min(stock_models)}-{max(stock_models)} per year, out of {t['symbol'].nunique()}")
    say(f"\n{'design':18s} {'mean AUC':>9s} {'years > A':>11s} {'filter dR':>10s} {'p':>7s} {'kept':>6s} verdict")
    rng = np.random.default_rng(0)
    a_auc = np.array(aucs["A_pooled"])
    tested = ~np.isnan(probs["A_pooled"])
    t, probs, thr = t[tested].reset_index(drop=True), {d: v[tested] for d, v in probs.items()}, {d: v[tested] for d, v in thr.items()}
    deltas = {}
    for d in DESIGNS:
        w = (probs[d] >= thr[d]).astype(float)
        obs, pval, _ = ml.block_bootstrap_p(t, w, rng)
        deltas[d] = obs
        beats = int((np.array(aucs[d]) > a_auc).sum())
        verdict = "" if d == "A_pooled" else ("HELPS" if beats >= 6 and obs > deltas["A_pooled"] else "no")
        say(f"{d:18s} {np.mean(aucs[d]):9.3f} {beats if d != 'A_pooled' else '-':>11} {obs:+10.4f} {pval:7.4f} "
            f"{int(w.sum()):6d} {verdict}")
    log.close()


if __name__ == "__main__":
    main()
