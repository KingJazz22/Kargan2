"""System 2: Alexander Elder's Triple Screen Trading System.

Screen 1 (tide, weekly): 13-period weekly EMA slope sets the allowed
direction -- rising EMA = only longs, falling EMA = only shorts.
Screen 2 (wave, daily): Force Index(2) dipping against the tide (negative in
an uptrend, positive in a downtrend) marks a pullback -- the setup.
Screen 3 (ripple, entry trigger): once the setup is active, entry fills at
the next daily bar's open (this project's usual next-bar-open convention).

Ported from OrlandMagics/elder_strategy.py. Unlike System 1, this one has a
real 10-year backtest sample (130-370 trades/asset), though even its best
cross-asset-generalizing exit rule (elder_exits.make_adx_fade(threshold=25))
is only barely positive -- see the plan/memory notes on why this was wired
in anyway.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from indicators_core import atr, ema

from .indicators_extra import force_index

WEEKLY_EMA_PERIOD = 13
FORCE_INDEX_PERIOD = 2
ATR_STOP_MULT = 2.0
TRAIL_ATR_MULT = 1.5
FIXED_TARGET_R = 2.0


@dataclass
class ElderSignal:
    direction: str  # "up" / "down"
    entry_price: float
    stop: float
    target: float
    risk: float


@dataclass
class ElderPosition:
    direction: str
    entry_price: float
    stop: float
    initial_stop: float
    entry_atr: float
    trail_active: bool = False
    extreme_price: float = 0.0

    def __post_init__(self):
        self.extreme_price = self.entry_price


def weekly_tide(weekly_df: pd.DataFrame) -> str:
    """'up' if the 13-week EMA is rising, 'down' if falling, 'none' otherwise."""
    if weekly_df is None or len(weekly_df) < WEEKLY_EMA_PERIOD + 2:
        return "none"
    ema13 = ema(weekly_df["Close"], WEEKLY_EMA_PERIOD)
    if pd.isna(ema13.iloc[-1]) or pd.isna(ema13.iloc[-2]):
        return "none"
    if ema13.iloc[-1] > ema13.iloc[-2]:
        return "up"
    if ema13.iloc[-1] < ema13.iloc[-2]:
        return "down"
    return "none"


def daily_setup(daily_df: pd.DataFrame, tide: str) -> bool:
    """True if today's Force Index(2) dips against the tide -- the 'wave' pullback setup."""
    if tide == "none" or daily_df is None or len(daily_df) < 30:
        return False
    fi = force_index(daily_df["Close"], daily_df["Volume"], FORCE_INDEX_PERIOD)
    if pd.isna(fi.iloc[-1]):
        return False
    return fi.iloc[-1] < 0 if tide == "up" else fi.iloc[-1] > 0


def check_entry(daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> Optional[ElderSignal]:
    tide = weekly_tide(weekly_df)
    if tide == "none":
        return None
    if not daily_setup(daily_df, tide):
        return None

    entry_price = float(daily_df["Close"].iloc[-1])
    atr_val = float(atr(daily_df["High"], daily_df["Low"], daily_df["Close"]).iloc[-1])
    if atr_val <= 0:
        return None

    stop_distance = ATR_STOP_MULT * atr_val
    stop = entry_price - stop_distance if tide == "up" else entry_price + stop_distance

    risk = abs(entry_price - stop)
    if risk <= 0:
        return None
    target = entry_price + FIXED_TARGET_R * risk if tide == "up" else entry_price - FIXED_TARGET_R * risk

    return ElderSignal(direction=tide, entry_price=entry_price, stop=stop, target=target, risk=risk)


def open_position(signal: ElderSignal, atr_val: float) -> ElderPosition:
    return ElderPosition(direction=signal.direction, entry_price=signal.entry_price,
                          stop=signal.stop, initial_stop=signal.stop, entry_atr=atr_val)
