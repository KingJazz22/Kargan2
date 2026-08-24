"""System 1: entry combiner (MTF sync + W-formation) + swappable exit rules.

Entry fires only when both required confirmations agree on direction:
  1. mtf_score -> MTF sync: 5m and 15m both trending the same direction on
     their normal (candlestick) charts, corroborated by levels (not boxed in
     a consolidation zone on the daily chart)
  2. pattern_w -> W-formation (double bottom) / M-formation (double top) on a
     Renko chart built from the 5m candles, tried at multiple brick sizes
     (50/25/10% of ATR)

Ported from OrlandMagics/strategy.py. The trident (three-push) confirmation
and its `require_trident` option were dropped in this port -- OrlandMagics
runs with trident off by default (`require_trident=False`), and its own
ablation testing found trident rarely fires on real 5m data and made no
difference to trade frequency, so porting that code would just be dead
weight here.

Every detector, and every regime input, only ever sees closed candles --
enforced by whatever fetches timeframe_data (see data_fetch_alpaca.py),
not by this module.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from indicators_core import atr

from . import mtf_score as mtf_score_mod
from . import pattern_w
from . import regime_filter

ATR_STOP_BUFFER_MULT = 0.1
TRAIL_ACTIVATE_R = 1.0
TRAIL_ATR_MULT = 1.5
FIXED_TARGET_R = 2.0
FALLBACK_ATR_STOP_MULT = 2.0

# W-formation is timed off the 5m chart -- that's the finest timeframe
# available and gives the 10-minute-level precision alerts used to use.
PATTERN_TIMEFRAME = "5m"

# Renko/S-R pattern detection only needs recent structure -- swings from long
# ago are noise, not signal. Bounding the lookback also keeps a day-by-day
# backtest linear instead of quadratic.
PATTERN_LOOKBACK_BARS = 300


def trim_for_patterns(df):
    return df.tail(PATTERN_LOOKBACK_BARS) if df is not None else df


@dataclass
class Signal:
    symbol: str
    direction: str
    entry_price: float
    stop: float
    target: float
    risk: float
    mtf_score: float
    intensity: str
    confirmations: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    stop: float
    initial_stop: float
    entry_atr: float
    trail_active: bool = False
    extreme_price: float = 0.0

    def __post_init__(self):
        self.extreme_price = self.entry_price


def compute_regimes(timeframe_data: dict) -> dict:
    return {tf: regime_filter.latest_regime(df) for tf, df in timeframe_data.items()
            if df is not None and len(df) > 0}


def check_entry(symbol: str, timeframe_data: dict, levels_by_tf: dict,
                 require_w: bool = True) -> Optional[Signal]:
    """timeframe_data: {timeframe: OHLCV df}. levels_by_tf: {timeframe: levels.compute_levels(df)}."""
    regimes = compute_regimes(timeframe_data)
    if len(regimes) < len(timeframe_data):
        return None  # missing data for a timeframe -- don't guess, skip this cycle

    mtf = mtf_score_mod.compute_mtf_score(regimes, symbol=symbol)
    if not mtf.confirmed:
        return None

    daily_df = timeframe_data.get("1D")
    if daily_df is None or len(daily_df) < 30:
        return None
    daily_levels = levels_by_tf.get("1D", {})
    if daily_levels.get("in_consolidation_zone"):
        return None

    pattern_df = timeframe_data.get(PATTERN_TIMEFRAME)
    if pattern_df is None or len(pattern_df) < 30:
        return None
    recent_pattern = trim_for_patterns(pattern_df)

    w_ts = None
    if require_w:
        w_direction, w_ts, _w_atr_mult = pattern_w.find_w_formation_multi(recent_pattern)
        if w_direction != mtf.direction or w_ts is None:
            return None

    entry_price = float(pattern_df["Close"].iloc[-1])
    atr_val = float(atr(pattern_df["High"], pattern_df["Low"], pattern_df["Close"]).iloc[-1])
    if atr_val <= 0:
        return None

    stop_distance = FALLBACK_ATR_STOP_MULT * atr_val
    if mtf.direction == "up":
        structural_stop = entry_price - stop_distance
        sr_stop = daily_levels.get("support")
        if sr_stop is not None:
            structural_stop = max(structural_stop, sr_stop)
    else:
        structural_stop = entry_price + stop_distance
        sr_stop = daily_levels.get("resistance")
        if sr_stop is not None:
            structural_stop = min(structural_stop, sr_stop)

    risk = abs(entry_price - structural_stop)
    if risk <= 0:
        return None
    target = (entry_price + FIXED_TARGET_R * risk if mtf.direction == "up"
              else entry_price - FIXED_TARGET_R * risk)

    return Signal(
        symbol=symbol,
        direction=mtf.direction,
        entry_price=entry_price,
        stop=structural_stop,
        target=target,
        risk=risk,
        mtf_score=mtf.score,
        intensity=mtf.intensity,
        confirmations={
            "mtf": True,
            "mtf_combo": mtf.matched_combo,
            "w_formation": require_w,
            "w_timestamp": str(w_ts) if w_ts is not None else None,
        },
    )


def open_position(signal: Signal, atr_val: float) -> Position:
    return Position(
        symbol=signal.symbol,
        direction=signal.direction,
        entry_price=signal.entry_price,
        stop=signal.stop,
        initial_stop=signal.stop,
        entry_atr=atr_val,
    )


# ---------------------------------------------------------------------------
# Exit rule candidates. Common signature: rule(position, bar, context) -> reason|None
# `bar` is a dict-like with Open/High/Low/Close. `context` may carry
# atr_val, levels (dict from levels.compute_levels), and mtf (MTFResult).
# Rules may mutate position.stop / position.trail_active / position.extreme_price.
# ---------------------------------------------------------------------------

def exit_atr_trail(position: Position, bar, context: dict) -> Optional[str]:
    atr_val = context.get("atr_val", position.entry_atr)
    close = bar["Close"]
    initial_risk = abs(position.entry_price - position.initial_stop)

    if position.direction == "up":
        position.extreme_price = max(position.extreme_price, close)
        profit = position.extreme_price - position.entry_price
        if profit >= TRAIL_ACTIVATE_R * initial_risk:
            position.trail_active = True
        if position.trail_active:
            position.stop = max(position.stop, position.extreme_price - TRAIL_ATR_MULT * atr_val)
        exit_now = bar["Low"] <= position.stop
    else:
        position.extreme_price = min(position.extreme_price, close)
        profit = position.entry_price - position.extreme_price
        if profit >= TRAIL_ACTIVATE_R * initial_risk:
            position.trail_active = True
        if position.trail_active:
            position.stop = min(position.stop, position.extreme_price + TRAIL_ATR_MULT * atr_val)
        exit_now = bar["High"] >= position.stop

    return "stop" if exit_now else None


def exit_structural_sr(position: Position, bar, context: dict) -> Optional[str]:
    lv = context.get("levels", {})
    if position.direction == "up":
        sr = lv.get("support")
        if sr is not None:
            position.stop = max(position.stop, sr)
        exit_now = bar["Low"] <= position.stop
    else:
        sr = lv.get("resistance")
        if sr is not None:
            position.stop = min(position.stop, sr)
        exit_now = bar["High"] >= position.stop
    return "stop" if exit_now else None


def exit_regime_flip(position: Position, bar, context: dict) -> Optional[str]:
    """Flatten the moment the MTF score drops out of confirmation or price
    re-enters a consolidation zone -- direct enforcement of never holding
    through consolidation, regardless of stop distance."""
    mtf = context.get("mtf")
    lv = context.get("levels", {})

    if mtf is not None and (not mtf.confirmed or mtf.direction != position.direction):
        return "regime_flip"
    if lv.get("in_consolidation_zone"):
        return "consolidation_zone"

    if position.direction == "up" and bar["Low"] <= position.stop:
        return "stop"
    if position.direction == "down" and bar["High"] >= position.stop:
        return "stop"
    return None


def exit_fixed_r(position: Position, bar, context: dict, r_multiple: float = FIXED_TARGET_R) -> Optional[str]:
    risk = abs(position.entry_price - position.initial_stop)
    target = (position.entry_price + r_multiple * risk if position.direction == "up"
              else position.entry_price - r_multiple * risk)

    if position.direction == "up":
        if bar["High"] >= target:
            return "target"
        if bar["Low"] <= position.stop:
            return "stop"
    else:
        if bar["Low"] <= target:
            return "target"
        if bar["High"] >= position.stop:
            return "stop"
    return None


def exit_regime_flip_plus_atr_trail(position: Position, bar, context: dict) -> Optional[str]:
    """Regime-flip/consolidation as a hard override on top of the ATR trail."""
    mtf = context.get("mtf")
    lv = context.get("levels", {})
    if mtf is not None and (not mtf.confirmed or mtf.direction != position.direction):
        return "regime_flip"
    if lv.get("in_consolidation_zone"):
        return "consolidation_zone"
    return exit_atr_trail(position, bar, context)


EXIT_RULES: dict[str, Callable[[Position, dict, dict], Optional[str]]] = {
    "atr_trail": exit_atr_trail,
    "structural_sr": exit_structural_sr,
    "regime_flip": exit_regime_flip,
    "fixed_r": exit_fixed_r,
    "regime_flip_plus_atr_trail": exit_regime_flip_plus_atr_trail,
}
