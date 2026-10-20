"""ML flat-market long-short (Kargan2 V3): features, panel, model, selection.

Shared by backtest_ml_flat.py (walk-forward research) and live_trading_v3.py
(paper bot), so live features are computed by exactly the backtested code.
Only depends on pandas/numpy/sklearn + strategy_range_fade/indicators_core --
no broker or research-cache imports, so it runs on Railway as-is.

See FINDINGS_ML_FLAT.md: universe = stocks whose own ADX(14) < 25 on each of
the last 10 bars; target = 5-day forward excess return vs SPY, ranked per
date; live model = Ridge, long top K / short bottom K.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import strategy_range_fade as rf
from indicators_core import rsi

ETFS = {"SPY", "QQQ", "IWM", "DIA"}
HORIZON = 5
EMBARGO = 10
RIDGE_ALPHA = 10.0
TOP_K = 5
# FI's history before this date is a different security (ticker reuse) -- see split_adjust.py.
FI_START = pd.Timestamp("2023-06-07")

# The 100 non-ETF symbols the backtest was run on (TOP100 + live universe).
UNIVERSE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "AMAT", "AMD", "AMGN", "AMZN", "AON", "APD",
    "AVGO", "BA", "BAC", "BDX", "BKNG", "BLK", "BSX", "C", "CAT", "CB", "CI", "COP", "COST", "CRM",
    "CSCO", "CVS", "CVX", "DE", "DIS", "DUK", "ELV", "EQIX", "ETN", "FI", "GILD", "GOOGL", "GS", "HD",
    "HON", "IBM", "INTC", "INTU", "ISRG", "ITW", "JNJ", "JPM", "KLAC", "KO", "LIN", "LLY", "LMT", "LOW",
    "MCD", "MDLZ", "MDT", "META", "MMC", "MRK", "MS", "MSFT", "MU", "NFLX", "NKE", "NOW", "NVDA", "ORCL",
    "PANW", "PEP", "PFE", "PG", "PGR", "PLD", "PM", "QCOM", "REGN", "RTX", "SBUX", "SCHW", "SLB", "SO",
    "SPGI", "SYK", "T", "TJX", "TMO", "TSLA", "TXN", "UNH", "UNP", "UPS", "V", "VRTX", "VZ", "WFC",
    "WMT", "XOM", "ZTS",
]
NON_FEATURES = ("date", "symbol", "fwd", "fwd_spy", "xfwd", "target")


def norm_index(df):
    df = df.copy()
    df.index = (df.index.tz_convert(None) if df.index.tz is not None else df.index).normalize()
    return df[~df.index.duplicated()].sort_index()


def symbol_features(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    p = rf.prepare(df, regime="adx", trigger="any")
    c, o = p["Close"], p["Open"]
    f = pd.DataFrame(index=p.index)
    spy_c = spy["Close"].reindex(p.index)
    for n in (1, 3, 5, 10, 20):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
        f[f"xret_{n}"] = f[f"ret_{n}"] - np.log(spy_c / spy_c.shift(n))
    bb_std = c.rolling(rf.BB_PERIOD).std()
    f["bb_z"] = (c - p["bb_mid"]) / bb_std
    f["rsi2"] = p["rsi2"]
    f["rsi14"] = rsi(c, 14)
    f["adx"] = p["adx"]
    ema34 = c.ewm(span=rf.EMA_PERIOD, adjust=False).mean()
    f["ema_slope"] = (ema34 - ema34.shift(rf.SLOPE_BARS)) / (rf.SLOPE_BARS * p["atr"])
    f["atr_pct"] = p["atr"] / c
    f["vol_ratio"] = p["Volume"] / p["Volume"].rolling(20).mean()
    lr = np.log(c).diff()
    f["rv_ratio"] = lr.rolling(20).std() / lr.rolling(60).std()
    f["dist_hi20"] = (p["High"].rolling(20).max() - c) / p["atr"]
    f["dist_lo20"] = (c - p["Low"].rolling(20).min()) / p["atr"]
    f["flat"] = p["regime_flat"]
    spy_o = spy["Open"].reindex(p.index)
    f["fwd"] = o.shift(-(HORIZON + 1)) / o.shift(-1) - 1
    f["fwd_spy"] = spy_o.shift(-(HORIZON + 1)) / spy_o.shift(-1) - 1
    return f


def build_panel(cache: dict) -> pd.DataFrame:
    """cache: {symbol: split-adjusted daily OHLCV}, must include SPY.
    Returns one row per flat stock-day with features (+ fwd/target where known)."""
    spy = norm_index(cache["SPY"])
    sp = rf.prepare(spy, regime="adx", trigger="any")
    spy_feats = pd.DataFrame({
        "spy_ret5": np.log(spy["Close"] / spy["Close"].shift(5)),
        "spy_adx": sp["adx"],
        "spy_bb_z": (spy["Close"] - sp["bb_mid"]) / spy["Close"].rolling(rf.BB_PERIOD).std(),
        "spy_flat": sp["regime_flat"].astype(float),
    })
    frames = []
    for s, df in cache.items():
        if s in ETFS or df is None or len(df) < 300:
            continue
        df = norm_index(df)
        if s == "FI":
            df = df[df.index >= FI_START]
            if len(df) < 300:
                continue
        f = symbol_features(df, spy)
        f["symbol"] = s
        frames.append(f)
    panel = pd.concat(frames).join(spy_feats, how="left")
    panel.index.name = "date"
    panel = panel.reset_index()
    panel = panel[panel["flat"].astype(bool)].drop(columns="flat")
    panel = panel.dropna(subset=[c for c in panel.columns if c not in ("fwd", "fwd_spy")])
    panel["xfwd"] = panel["fwd"] - panel["fwd_spy"]
    g = panel.groupby("date")
    panel["cs_rank_ret5"] = g["ret_5"].rank(pct=True)
    panel["cs_rank_bbz"] = g["bb_z"].rank(pct=True)
    panel["n_flat"] = g["symbol"].transform("count")
    panel["target"] = g["xfwd"].rank(pct=True)
    return panel


def feature_columns(panel: pd.DataFrame):
    return [c for c in panel.columns if c not in NON_FEATURES]


def training_rows(panel: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Rows usable for training a model applied from `before` on: entry date at
    least EMBARGO trading days earlier and a fully realized forward return."""
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    pos = days.searchsorted(before)
    cutoff = days[max(0, pos - EMBARGO)]
    return panel[(panel["date"] < cutoff) & panel["target"].notna() & panel["fwd"].notna()]


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def select_long_short(scored: pd.DataFrame, k: int = TOP_K):
    """scored: one date's flat stocks with a `score` column. Returns (longs, shorts)."""
    g = scored.sort_values("score")
    return list(g["symbol"].tail(k))[::-1], list(g["symbol"].head(k))
