"""XGBoost vs random forest meta-labeler on all swing_pcse_exit trades, per
research_metalabel/SPEC_XGB.md: same yearly walk-forward as
backtest_rf_grouped.py (train only on trades closed before each test year),
feature sets A (16 base) and B (base + liquidity, sector, shrunk stock loss
rate). Writes research_metalabel/results_xgb.txt.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import backtest_metalabel_pcse as ml
import backtest_rf_grouped as grp

XGB = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=10, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", random_state=0, n_jobs=-1,
                    eval_metric="logloss")
MODELS = ["rf_A", "rf_B", "xgb_A", "xgb_B"]


def fit_predict(kind, X, y, tr, te):
    if kind == "rf":
        m = clone(grp.RF)
    else:
        m = clone(XGB).set_params(scale_pos_weight=(y[tr] == 0).sum() / (y[tr] == 1).sum())
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[tr])[:, 1], m.predict_proba(X[te])[:, 1]


def main():
    log = open(ml.OUT_DIR / "results_xgb.txt", "w", encoding="utf-8")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")

    t = pd.read_pickle("_pcse_fresh_trades.pkl").sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    t = grp.group_features(t)
    y = t["y"].to_numpy()
    Xa = t[ml.FEATURES].to_numpy()
    sector_oh = pd.get_dummies(t["sector"]).to_numpy(float)
    stk = t["symbol"].to_numpy()
    entry_year = t["entry_time"].dt.year.to_numpy()

    probs = {m: np.full(len(t), np.nan) for m in MODELS}
    thr = {m: np.full(len(t), np.nan) for m in MODELS}
    aucs = {m: [] for m in MODELS}
    for year in grp.TEST_YEARS:
        tr = np.where(t["exit_time"] < pd.Timestamp(f"{year}-01-01"))[0]
        te = np.where(entry_year == year)[0]
        g_rate = 1 - y[tr].mean()
        st = pd.DataFrame({"s": stk[tr], "loss": 1 - y[tr]}).groupby("s")["loss"].agg(["sum", "count"])
        stock_rate = pd.Series(stk).map((st["sum"] + grp.LOSS_PRIOR * g_rate) / (st["count"] + grp.LOSS_PRIOR)) \
            .fillna(g_rate).to_numpy()
        Xb = np.column_stack([Xa, t["liq_pct"].fillna(0.5).to_numpy(), sector_oh, stock_rate])
        for m in MODELS:
            kind, fs = m.split("_")
            p_in, p_te = fit_predict(kind, Xa if fs == "A" else Xb, y, tr, te)
            aucs[m].append(roc_auc_score(y[te], p_te))
            probs[m][te] = p_te
            thr[m][te] = np.quantile(p_in, ml.FILTER_Q)
        say(f"  {year} (train {len(tr)}, test {len(te)}): " + " ".join(f"{m}={aucs[m][-1]:.3f}" for m in MODELS))

    tested = ~np.isnan(probs["rf_A"])
    tt = t[tested].reset_index(drop=True)
    rng = np.random.default_rng(0)
    say(f"\n{'model':7s} {'mean AUC':>9s} {'filter dR':>10s} {'p':>7s} {'kept':>6s} {'losers skipped':>15s}")
    delta = {}
    for m in MODELS:
        w = (probs[m][tested] >= thr[m][tested]).astype(float)
        obs, pval, _ = ml.block_bootstrap_p(tt, w, rng)
        delta[m] = obs
        lost = ((w == 0) & (tt["y"].to_numpy() == 0)).sum()
        say(f"{m:7s} {np.mean(aucs[m]):9.3f} {obs:+10.4f} {pval:7.4f} {int(w.sum()):6d} "
            f"{lost:7d}/{(tt['y'] == 0).sum()}")
    beats_b = int((np.array(aucs["xgb_B"]) > np.array(aucs["rf_B"])).sum())
    beats_a = int((np.array(aucs["xgb_A"]) > np.array(aucs["rf_A"])).sum())
    replace = beats_b >= 6 and delta["xgb_B"] > delta["rf_B"]
    say(f"\nxgb_B beats rf_B AUC in {beats_b}/8 years (needs >= 6); filter dR {delta['xgb_B']:+.4f} vs "
        f"{delta['rf_B']:+.4f} -> {'REPLACE rf_B in the shadow log' if replace else 'keep rf_B'}")
    say(f"(info) xgb_A beats rf_A AUC in {beats_a}/8 years")
    log.close()


if __name__ == "__main__":
    main()
