"""Breakout Hunter strategy: trade breakouts that follow a genuine
volatility contraction ("squeeze"). The original spec never mentioned a
breakdown/short leg (unlike the trend-following and mean-reversion specs,
which explicitly said "short is the exact opposite"), so none was added
originally, and `long_entry` / the FINDINGS_BREAKOUT.md conclusions are
unchanged.

`short_entry` was added later (for the grid-martingale strategy, which wants
to trade both directions) as the exact mirror of `long_entry`: squeeze
recently, close below `support` (prior 20-bar low) instead of above
`resistance`, same volume/ADX-rising confirmation. It has NOT been
backtested/validated the way the long side has -- per FINDINGS_MTF.md, this
project has direct precedent for a short side being a confirmed loser even
when the long side has a real edge, so this must not be assumed profitable
by symmetry. See FINDINGS_GRID_MARTINGALE.md for the actual result.

Rules (as specified, with gaps filled in and flagged):
  Detect (squeeze):  ATR in the bottom quartile of its own recent range, AND
                      volume below its own 20-bar average, AND Bollinger Band
                      width in the bottom quartile of its own recent range,
                      AND Keltner Channel width likewise in its own bottom
                      quartile. Bollinger and Keltner squeeze are read as
                      parallel, independent checks (each indicator's own
                      width vs. its own recent history), matching how the
                      spec lists them as separate bullet points -- not the
                      classic TTM cross-indicator "BB inside KC" condition,
                      which turned out to be a near-never-fires event on this
                      data (2.4% of bars) that killed the signal entirely.
  Entry:              close breaks above resistance (defined as the prior
                      20-bar high, excluding the breakout bar itself) AND
                      volume is above its own 20-bar average (same standard
                      used for "volume confirmation" in the trend-following
                      spec) AND ADX is rising (higher than 3 bars ago).

  Timing fix (same class of issue found in both earlier strategies): "low
  volume" and "high volume" cannot both be true on the breakout bar itself --
  by definition a breakout should occur ON expanding volume. So the squeeze
  conditions are checked over a RECENT lookback window (was there a genuine
  compression within the last 10 bars), not on the breakout bar itself, while
  the breakout/volume/ADX conditions are checked on the current bar.

  Gap filled in: no stop-loss was specified beyond "ATR trailing stop." Unlike
  the mean-reversion and multi-timeframe strategies (where the trail is active
  from bar 1), this is conceptually a trend-continuation trade -- breakouts
  often get an immediate throwback test of the broken level, so it reuses the
  ORIGINAL trend-following strategy's two-stage stop: a structural initial
  stop below the pre-breakout consolidation low, then the ATR trail only takes
  over once the trade has 1x ATR of profit cushion (see backtest_breakout.py).
"""
import pandas as pd

from indicators import adx, atr, bollinger_bands, keltner_channel, volume_sma

ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
KC_EMA_PERIOD = 20
KC_ATR_PERIOD = 10
KC_MULT = 1.5
ADX_PERIOD = 14
VOL_PERIOD = 20

PERCENTILE_LOOKBACK = 100
LOW_ATR_PERCENTILE = 0.25
LOW_BB_WIDTH_PERCENTILE = 0.25
LOW_KC_WIDTH_PERCENTILE = 0.25
SQUEEZE_LOOKBACK = 10

RESISTANCE_LOOKBACK = 20
HIGH_VOLUME_MULT = 1.0
ADX_RISING_LOOKBACK = 3

INITIAL_STOP_ATR_BUFFER = 0.5
TRAIL_ACTIVATE_ATR = 1.0
TRAIL_ATR_MULT = 2.5
EMERGENCY_STOP_BUFFER_ATR = 2.0  # beyond initial_stop, not independently from entry -- see backtest_breakout.py


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)
    out["adx"] = adx(out["High"], out["Low"], out["Close"], ADX_PERIOD)
    out["vol_sma"] = volume_sma(out["Volume"], VOL_PERIOD)

    bb_mid, bb_upper, bb_lower = bollinger_bands(out["Close"], BB_PERIOD, BB_STD)
    kc_mid, kc_upper, kc_lower = keltner_channel(out["High"], out["Low"], out["Close"], KC_EMA_PERIOD, KC_ATR_PERIOD, KC_MULT)
    bb_width = (bb_upper - bb_lower) / bb_mid
    kc_width = (kc_upper - kc_lower) / kc_mid

    atr_pct = _rolling_percentile_rank(out["atr"], PERCENTILE_LOOKBACK)
    bb_width_pct = _rolling_percentile_rank(bb_width, PERCENTILE_LOOKBACK)
    kc_width_pct = _rolling_percentile_rank(kc_width, PERCENTILE_LOOKBACK)

    low_atr = atr_pct <= LOW_ATR_PERCENTILE
    low_volume = out["Volume"] < out["vol_sma"]
    bb_squeeze = bb_width_pct <= LOW_BB_WIDTH_PERCENTILE
    kc_squeeze = kc_width_pct <= LOW_KC_WIDTH_PERCENTILE

    squeeze_now = low_atr & low_volume & bb_squeeze & kc_squeeze
    squeeze_recently = squeeze_now.rolling(SQUEEZE_LOOKBACK).max().astype(bool)

    out["resistance"] = out["High"].shift(1).rolling(RESISTANCE_LOOKBACK).max()
    out["support"] = out["Low"].shift(1).rolling(RESISTANCE_LOOKBACK).min()

    breaks_resistance = out["Close"] > out["resistance"]
    breaks_support = out["Close"] < out["support"]
    high_volume = out["Volume"] > HIGH_VOLUME_MULT * out["vol_sma"]
    adx_rising = out["adx"] > out["adx"].shift(ADX_RISING_LOOKBACK)

    out["long_entry"] = (squeeze_recently & breaks_resistance & high_volume & adx_rising).fillna(False)
    out["short_entry"] = (squeeze_recently & breaks_support & high_volume & adx_rising).fillna(False)

    return out
