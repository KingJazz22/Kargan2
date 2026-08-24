"""Per-timeframe trend-vs-consolidation regime classification.

Three independent conditions must agree before a bar counts as trending:
  1. Trend strength (Wilder ADX, tiered strong/medium/consolidation with hysteresis)
  2. Squeeze veto (BB/KC width percentile rank -- tight width forces consolidation)
  3. Directional bias (price vs. EMA(100))

Output per bar is one of:
  trending_up_strong / trending_up_medium /
  trending_down_strong / trending_down_medium / consolidation

Ported from OrlandMagics/regime_filter.py.
"""
import pandas as pd

from indicators_core import adx, bollinger_bands, ema, keltner_channel

from .indicators_extra import rolling_percentile_rank

ADX_STRONG = 35.0
ADX_MEDIUM = 25.0
ADX_CONSOLIDATION = 20.0  # below this, always consolidation (below the hysteresis band)
SQUEEZE_PERCENTILE = 0.20
TREND_LINE_PERIOD = 100

CONSOLIDATION = "consolidation"


def _tier_from_adx(adx_val: float, prev_tier: str) -> str:
    """Return 'strong' / 'medium' / 'consolidation' for one ADX reading, with
    the 20-25 band holding the previous tier (hysteresis) to avoid flicker."""
    if pd.isna(adx_val):
        return CONSOLIDATION
    if adx_val > ADX_STRONG:
        return "strong"
    if adx_val > ADX_MEDIUM:
        return "medium"
    if adx_val <= ADX_CONSOLIDATION:
        return CONSOLIDATION
    # 20 < adx_val <= 25: hysteresis band, hold last non-consolidation tier
    return prev_tier if prev_tier != CONSOLIDATION else CONSOLIDATION


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Given an OHLCV DataFrame, return a DataFrame indexed the same way with
    columns: adx, bb_width_pct, kc_width_pct, direction, tier, regime."""
    high, low, close = df["High"], df["Low"], df["Close"]

    adx_val = adx(high, low, close, 14)

    _, bb_upper, bb_lower = bollinger_bands(close, 20)
    bb_width = (bb_upper - bb_lower) / close
    bb_width_pct = rolling_percentile_rank(bb_width, 100)

    _, kc_upper, kc_lower = keltner_channel(high, low, close)
    kc_width = (kc_upper - kc_lower) / close
    kc_width_pct = rolling_percentile_rank(kc_width, 100)

    trend_line = ema(close, TREND_LINE_PERIOD)
    direction = pd.Series("up", index=df.index)
    direction[close < trend_line] = "down"

    squeezed = (bb_width_pct < SQUEEZE_PERCENTILE) | (kc_width_pct < SQUEEZE_PERCENTILE)

    tiers = []
    prev_tier = CONSOLIDATION
    for i in range(len(df)):
        if bool(squeezed.iloc[i]):
            tier = CONSOLIDATION
        else:
            tier = _tier_from_adx(adx_val.iloc[i], prev_tier)
        tiers.append(tier)
        prev_tier = tier
    tier_series = pd.Series(tiers, index=df.index)

    regime = pd.Series(CONSOLIDATION, index=df.index)
    trending_mask = tier_series != CONSOLIDATION
    regime[trending_mask] = (
        "trending_" + direction[trending_mask] + "_" + tier_series[trending_mask]
    )

    return pd.DataFrame({
        "adx": adx_val,
        "bb_width_pct": bb_width_pct,
        "kc_width_pct": kc_width_pct,
        "direction": direction,
        "tier": tier_series,
        "regime": regime,
    })


def latest_regime(df: pd.DataFrame) -> str:
    """Convenience: classify the whole series and return the last bar's regime."""
    classified = classify_regime(df)
    if classified.empty:
        return CONSOLIDATION
    return classified["regime"].iloc[-1]
