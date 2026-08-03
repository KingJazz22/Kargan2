"""Probabilistic Candle-State Edge (PCSE): entries come purely from
indicators_candle's OHLCV-derived features + model_candle_prob's walk-forward
Markov/logistic probability blend -- no canned indicators. Exit is a
confidence-adaptive trailing stop: distance and activation are set from the
trade's own entry confidence and the symbol's own rolling return-stdev
(never a fixed ATR multiple), plus a probability-decay early exit and an
R-multiple ratchet that tightens the trail as a winner runs.

prepare(df, ...) follows this repo's stateless convention: pure function of
a raw OHLCV DataFrame -> DataFrame with every column backtest_candle_prob.py
needs, broker-agnostic. See FINDINGS_CANDLE_PROB.md for every filled-in gap.
"""
import numpy as np
import pandas as pd

import indicators_candle as ic
import model_candle_prob as model

HORIZON = 5                 # bars ahead the model predicts over
COST_THRESHOLD = 0.0008     # round-trip cost hurdle baked into the label (~8bps)
TRAIN_BARS = 1500           # walk-forward trailing training window
TEST_BARS = 250             # walk-forward fold size (retrain cadence)

EDGE_MARGIN = 0.04          # both layers must beat the fold's own base rate by this many
                            # probability points to enter -- an asymmetric threshold label's
                            # unconditional base rate is rarely near 50%, so an ABSOLUTE cutoff
                            # like "p >= 0.55" is meaningless without knowing the base rate first
DECAY_HALF_LIFE_FRAC = 0.5  # exit early once in-trade edge decays below DECAY_HALF_LIFE_FRAC * EDGE_MARGIN

TRAIL_ACTIVATE_MULT = 1.0    # trail engages once profit >= this many entry-stdev units
HARD_STOP_MULT = 2.5         # initial hard stop, in entry-stdev units, active before the trail engages
TRAIL_MULT_LOW = 1.5         # trail distance (stdev units) at the minimum tradeable confidence
TRAIL_MULT_HIGH = 3.0        # trail distance (stdev units) at confidence == 1.0
RATCHET_LEVELS = [1.0, 2.0, 3.0]      # R-multiple milestones
RATCHET_TRAIL_CAPS = [2.5, 1.8, 1.2]  # trail distance capped at this x the ORIGINAL ("1R") stop distance, per milestone

LONG_ONLY = True   # short defaults OFF until independently validated -- see FINDINGS_MTF.md precedent


def _combined_confidence(p_markov: pd.Series, n_markov: pd.Series, p_logit: pd.Series,
                          base_rate: pd.Series, min_samples: int, edge_margin: float):
    """Both layers must independently beat the fold's own unconditional base
    rate by edge_margin, and the Markov side must have enough historical
    sample support -- a conservative "both agree" gate, not an average that
    lets one weak layer get pulled up by the other. "Confidence" here is the
    weaker (min) of the two layers' EDGE over base rate, not a raw
    probability -- comparable across folds/timeframes/cost thresholds even
    though the base rate itself moves around a lot between them."""
    ok_markov = (n_markov >= min_samples) & (p_markov - base_rate >= edge_margin)
    ok_logit = (p_logit - base_rate) >= edge_margin
    entry = (ok_markov & ok_logit).fillna(False)
    confidence = np.minimum(p_markov, p_logit) - base_rate
    return entry, confidence


def prepare(df: pd.DataFrame, *, horizon: int = HORIZON, cost_threshold: float = COST_THRESHOLD,
            train_bars: int = TRAIN_BARS, test_bars: int = TEST_BARS,
            edge_margin: float = EDGE_MARGIN, long_only: bool = LONG_ONLY,
            min_samples: int = model.MIN_STATE_SAMPLES) -> pd.DataFrame:
    feat = ic.build_features(df)
    preds = model.walk_forward_predict(
        feat, train_bars=train_bars, test_bars=test_bars,
        horizon=horizon, cost_threshold=cost_threshold, min_samples=min_samples,
    )
    out = feat.join(preds)

    long_ok, long_conf = _combined_confidence(
        out["p_long_markov"], out["n_long_markov"], out["p_long_logit"],
        out["base_rate_long"], min_samples, edge_margin)
    out["long_entry"] = long_ok
    out["long_confidence"] = long_conf

    if long_only:
        out["short_entry"] = False
        out["short_confidence"] = np.nan
    else:
        short_ok, short_conf = _combined_confidence(
            out["p_short_markov"], out["n_short_markov"], out["p_short_logit"],
            out["base_rate_short"], min_samples, edge_margin)
        out["short_entry"] = short_ok
        out["short_confidence"] = short_conf

    return out


def confidence_to_trail_mult(confidence_edge: float, edge_margin: float = EDGE_MARGIN) -> float:
    """Linear map from edge-over-base-rate in [edge_margin, 3*edge_margin] to
    [TRAIL_MULT_LOW, TRAIL_MULT_HIGH] stop distance -- higher-edge entries
    get more room to run, marginal ones trail tight. The 3x soft cap is a
    filled-in gap (no natural upper bound on edge exists) -- see FINDINGS."""
    span = max(2.0 * edge_margin, 1e-6)
    frac = np.clip((confidence_edge - edge_margin) / span, 0.0, 1.0)
    return TRAIL_MULT_LOW + frac * (TRAIL_MULT_HIGH - TRAIL_MULT_LOW)
