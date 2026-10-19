"""Exports the training set for the live random-forest shadow scorer
(rf_shadow.py): design B from FINDINGS_RF_GROUPED.md -- one pooled RF on all
9,861 closed swing_pcse_exit trades, with liquidity percentile, inferred
sector and shrunk per-stock loss rate as extra inputs.

Writes models/rf_shadow_v1_train.parquet (features + label per trade) and
models/rf_shadow_v1_meta.json. The live side refits the forest from these
at startup instead of unpickling one, so a different scikit-learn version
on Railway can't break loading. Re-run only to cut a new model version.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_pcse_fresh_universe as fu
import backtest_rf_grouped as grp
from metalabel_features import FEATURES

VERSION = "rf_shadow_v1"
OUT = Path("models")


def liquidity_reference():
    """63-day median dollar volume of every stock with a bar on the last data day -- live signals
    are ranked against this to reproduce the training-time cross-sectional percentile."""
    bars = fu.load_bars()
    close = bars.pivot_table(index="date", columns="symbol", values="Close", aggfunc="last").sort_index()
    vol = bars.pivot_table(index="date", columns="symbol", values="Volume", aggfunc="last").reindex_like(close)
    dv = (close * vol).rolling(63, min_periods=40).median().iloc[-1].dropna()
    return close.index[-1], np.sort(dv.to_numpy())


def main():
    t = pd.read_pickle("_pcse_fresh_trades.pkl").sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    t = grp.group_features(t)
    train = t[["symbol", "entry_time"] + FEATURES + ["liq_pct", "sector", "y"]].copy()
    ref_date, ref = liquidity_reference()

    g_rate = float(1 - train["y"].mean())
    st = train.assign(loss=1 - train["y"]).groupby("symbol")["loss"].agg(["sum", "count"])
    shrunk = ((st["sum"] + grp.LOSS_PRIOR * g_rate) / (st["count"] + grp.LOSS_PRIOR)).round(5)

    OUT.mkdir(exist_ok=True)
    train.to_parquet(OUT / f"{VERSION}_train.parquet", index=False)
    meta = {
        "version": VERSION,
        "design": "B_pooled+groups (FINDINGS_RF_GROUPED.md)",
        "base_features": FEATURES,
        "sector_etfs": grp.SECTOR_ETFS,
        "sector_columns": sorted(train["sector"].unique()),
        "rf_params": grp.RF.get_params(),
        "filter_q": 0.20,
        "loss_prior": grp.LOSS_PRIOR,
        "global_loss_rate": g_rate,
        "stock_loss_rate": shrunk.to_dict(),
        "liq_reference_date": str(ref_date.date()),
        "liq_reference_dollar_volume": ref.round(0).tolist(),
        "n_trades": len(train), "n_losers": int((train["y"] == 0).sum()),
        "trained_through": str(train["entry_time"].max().date()),
    }
    meta["rf_params"].pop("n_jobs", None)
    (OUT / f"{VERSION}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"wrote {VERSION}: {len(train)} trades, {meta['n_losers']} losers, "
          f"{len(ref)} stocks in the liquidity reference ({ref_date.date()})")


if __name__ == "__main__":
    main()
