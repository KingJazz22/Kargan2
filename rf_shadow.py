"""Shadow random-forest scorer for live swing_pcse_exit signals.

READ-ONLY with respect to trading: it never places, skips or resizes an
order. For each live entry signal it records the RF's win probability and
whether the pre-registered filter (skip the bottom 20% by in-sample
probability) WOULD have skipped it, so the filter can be judged on 6-12
months of forward trades -- the one test the backtests can no longer give
(FINDINGS_RF_GROUPED.md). Every failure is printed and swallowed.

Model: design B of FINDINGS_RF_GROUPED.md, refit at first use from
models/<VERSION>_train.parquet + _meta.json (build_rf_shadow_model.py), not
unpickled, so a scikit-learn upgrade on the server can't break loading.
Caveat: live daily bars are split-UNadjusted; a split inside the last 60
bars distorts the return features for that one signal (flagged in the log).
"""
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from metalabel_features import feature_frame, spy_features

VERSION = "rf_shadow_v1"
MODEL_DIR = Path(__file__).resolve().parent / "models"
LOG_NAME = "rf_shadow_log_v2.csv"
LOG_FIELDS = ["logged_at", "model", "symbol", "signal_date", "entry_hint", "atr", "initial_stop", "take_profit",
              "prob_win", "threshold", "would_skip", "entries_halted", "sector", "liq_pct", "split_suspect"]

_model = None          # (rf, meta, threshold), fitted lazily
_etf_cache = {}        # signal_date -> {etf: df}, one fetch per run


def _design_matrix(base, liq_pct, sector, symbols, meta):
    cols = meta["sector_columns"]
    onehot = np.array([[1.0 if s == c else 0.0 for c in cols] for s in sector])
    rates = meta["stock_loss_rate"]
    stock_rate = np.array([rates.get(s, meta["global_loss_rate"]) for s in symbols])
    return np.column_stack([np.asarray(base, float), np.nan_to_num(np.asarray(liq_pct, float), nan=0.5),
                            onehot, stock_rate])


def _load():
    global _model
    if _model is None:
        from sklearn.ensemble import RandomForestClassifier
        meta = json.loads((MODEL_DIR / f"{VERSION}_meta.json").read_text())
        train = pd.read_parquet(MODEL_DIR / f"{VERSION}_train.parquet")
        X = _design_matrix(train[meta["base_features"]], train["liq_pct"], train["sector"], train["symbol"], meta)
        rf = RandomForestClassifier(**{**meta["rf_params"], "n_jobs": -1}).fit(X, train["y"].to_numpy())
        threshold = float(np.quantile(rf.predict_proba(X)[:, 1], meta["filter_q"]))
        _model = (rf, meta, threshold)
    return _model


def _sector(close, etf_bars, etfs):
    """SPDR sector ETF with the highest 252-day daily-return correlation (as in training)."""
    rets = pd.DataFrame({"_s": close}).join(
        pd.DataFrame({e: etf_bars[e]["Close"] for e in etfs if e in etf_bars}), how="left").pct_change(fill_method=None)
    window = rets.tail(252)
    ok = window["_s"].notna()
    if ok.sum() < 120:
        return "unknown"
    corr = window[ok].drop(columns="_s").corrwith(window["_s"][ok])
    return corr.idxmax() if corr.notna().any() else "unknown"


def score(symbol, prepared, entry_ts, spy_df, etf_bars):
    rf, meta, threshold = _load()
    p = prepared.loc[:entry_ts]
    row = feature_frame(p, spy_features(spy_df)).iloc[-1]
    dv = (p["Close"] * p["Volume"]).rolling(63, min_periods=40).median().iloc[-1]
    ref = np.asarray(meta["liq_reference_dollar_volume"])
    liq_pct = float(np.searchsorted(ref, dv, side="right") / len(ref)) if np.isfinite(dv) else np.nan
    sector = _sector(p["Close"], etf_bars, meta["sector_etfs"])
    X = _design_matrix([row[meta["base_features"]].to_numpy()], [liq_pct], [sector], [symbol], meta)
    prob = float(rf.predict_proba(X)[0, 1])
    gap = (p["Open"] / p["Close"].shift(1)).tail(60)
    return {"prob_win": round(prob, 4), "threshold": round(threshold, 4), "would_skip": prob < threshold,
            "sector": sector, "liq_pct": round(liq_pct, 3) if np.isfinite(liq_pct) else "",
            "split_suspect": bool(((gap < 0.55) | (gap > 1.8)).any())}


def log_signal(symbol, prepared, entry_ts, bars, fetch_daily, entry_hint, atr, initial_stop, take_profit,
               entries_halted, dry_run, state_dir):
    """Score one live signal and append it to the shadow log. Never raises."""
    try:
        key = str(entry_ts.date())
        if key not in _etf_cache:
            _etf_cache.clear()
            _etf_cache[key] = fetch_daily(_load()[1]["sector_etfs"])
        s = score(symbol, prepared, entry_ts, bars["SPY"], _etf_cache[key])
        verdict = "WOULD SKIP" if s["would_skip"] else "would take"
        print(f"    rf_shadow: p(win)={s['prob_win']:.3f} vs threshold {s['threshold']:.3f} -> {verdict} "
              f"[{s['sector']}, liq {s['liq_pct']}]{' (split suspected)' if s['split_suspect'] else ''} -- log only")
        if dry_run:
            return
        path = Path(state_dir) / LOG_NAME
        new = not path.exists()
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"logged_at": datetime.now().isoformat(timespec="seconds"), "model": VERSION,
                        "symbol": symbol, "signal_date": key, "entry_hint": round(entry_hint, 4),
                        "atr": round(atr, 4), "initial_stop": round(initial_stop, 4),
                        "take_profit": round(take_profit, 4), "entries_halted": entries_halted, **s})
    except Exception as e:  # shadow logging must never interfere with trading
        print(f"    rf_shadow: scoring failed for {symbol}: {e!r} (no effect on orders)")
