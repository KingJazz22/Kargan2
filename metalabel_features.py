"""Signal-time features for the swing_pcse_exit meta-labeler. Shared by the
research scripts (backtest_metalabel_pcse.py and friends) and the live shadow
scorer (rf_shadow.py), so a live signal is described exactly as the training
trades were. Every value uses only bars up to and including the signal bar.
"""
import numpy as np
import pandas as pd

from indicators_core import adx, bollinger_bands, rsi

FEATURES = ["atr_pct", "adx14", "rsi14", "bb_z20", "breakout_atr", "range_atr", "vol_ratio",
            "ret5", "ret20", "ret60", "dist_252h", "rv_ratio",
            "spy_ret20", "spy_above_sma200", "spy_rv20", "spy_adx14"]


def spy_features(spy):
    c = spy["Close"]
    r = c.pct_change()
    return pd.DataFrame({
        "spy_ret20": c / c.shift(20) - 1,
        "spy_above_sma200": (c > c.rolling(200).mean()).astype(float),
        "spy_rv20": r.rolling(20).std() * np.sqrt(252),
        "spy_adx14": adx(spy["High"], spy["Low"], c, 14),
    })


def feature_frame(prepared, spy_feat):
    """prepared: strategy_swing_breakout.prepare() output. Returns one row of FEATURES per bar."""
    p = prepared
    c = p["Close"]
    r = c.pct_change()
    mid, upper, _ = bollinger_bands(c, 20, 2.0)
    return pd.DataFrame({
        "atr_pct": p["atr"] / c,
        "adx14": adx(p["High"], p["Low"], c, 14),
        "rsi14": rsi(c, 14),
        "bb_z20": (c - mid) / ((upper - mid) / 2.0),
        "breakout_atr": (c - p["resistance"]) / p["atr"],
        "range_atr": (p["resistance"] - p["support"]) / p["atr"],
        "vol_ratio": p["Volume"] / p["vol_sma"],
        "ret5": c / c.shift(5) - 1, "ret20": c / c.shift(20) - 1, "ret60": c / c.shift(60) - 1,
        "dist_252h": c / p["High"].rolling(252, min_periods=60).max() - 1,
        "rv_ratio": r.rolling(20).std() / r.rolling(60).std(),
    }).join(spy_feat, how="left")
