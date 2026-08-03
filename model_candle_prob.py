"""Walk-forward probabilistic model for PCSE: a Markov/n-gram transition table
over discrete candle-direction states, blended with a logistic regression over
continuous candle-geometry features (indicators_candle.FEATURE_COLUMNS). Both
layers are fit ONLY on a trailing training window and evaluated on the
following, strictly-later window (walk-forward, rolled forward fold by fold)
-- never a single train/test split. That's the direct fix for the MTF
strategy's Yahoo-~2yr-window overfitting trap documented in FINDINGS_MTF.md:
a model that only ever saw one bull-market window looked "confirmed" and then
lost money once retested on deep multi-regime history.

Long and short are fit as two completely independent binary problems --
nothing here assumes a signal that predicts up-moves also predicts down-moves
(this project has twice independently confirmed that assumption is false for
MTF's short side).

The Markov layer's per-state win probability is NOT the raw empirical mean --
that blew up in testing (with ~8 dominant 3-candle states and a 1500-bar
training window, most states never accumulate a stable sample within one
fold, so a blunt "raw mean AND n >= 200" gate left almost nothing tradeable).
Instead each state gets a Beta-Binomial posterior (this repo's own
bayesian_tracker.BetaBinomialPosterior, reused from the grid-martingale
strategy) and the ENTRY signal is the posterior's lower credible bound, not
its mean -- a state seen only a handful of times has a wide posterior and a
low credible bound regardless of how lucky its raw win rate looks, so sample
size is handled by the statistics itself rather than a separate magic-number
cutoff.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bayesian_tracker import BetaBinomialPosterior
from indicators_candle import FEATURE_COLUMNS

MIN_STATE_SAMPLES = 20    # sanity floor only -- the credible-lower-bound gate does the real work
CREDIBLE_PCT = 0.20       # use the state's 20th-percentile posterior estimate, not its raw mean
LOGIT_C = 0.5             # L2 regularization strength (smaller = more regularized, fewer features -> low overfit risk)
MIN_LOGIT_SAMPLES = 500   # minimum labeled rows before fitting a logistic model at all


def make_labels(df: pd.DataFrame, horizon: int, cost_threshold: float) -> pd.DataFrame:
    """Forward `horizon`-bar return and two independent binary labels: did it
    clear the round-trip cost hurdle up, or down. NaN for the last `horizon`
    bars of df (no future bar available yet) -- these get dropped downstream,
    never filled from beyond df's own boundary."""
    fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    y_long = pd.Series(np.where(fwd_ret > cost_threshold, 1.0, 0.0), index=df.index)
    y_short = pd.Series(np.where(fwd_ret < -cost_threshold, 1.0, 0.0), index=df.index)
    y_long[fwd_ret.isna()] = np.nan
    y_short[fwd_ret.isna()] = np.nan
    return pd.DataFrame({"fwd_ret": fwd_ret, "y_long": y_long, "y_short": y_short})


@dataclass
class ModelBundle:
    markov_long: dict
    markov_short: dict
    logit_long: tuple | None   # (StandardScaler, LogisticRegression) or None if not fittable
    logit_short: tuple | None
    feature_cols: list
    min_samples: int
    base_rate_long: float      # unconditional P(y_long) in the training window -- entries are
    base_rate_short: float     # gated on edge OVER this, not an absolute probability cutoff,
                                # since an asymmetric threshold label's base rate is rarely near 50%


def _fit_markov(states: pd.Series, y: pd.Series, min_samples: int) -> dict:
    table = {}
    for state, vals in y.groupby(states):
        vals = vals.dropna()
        n = len(vals)
        if n < min_samples:
            continue
        wins = int(vals.sum())
        posterior = BetaBinomialPosterior(alpha=1.0 + wins, beta=1.0 + (n - wins))
        table[state] = (posterior.credible_lower(pct=CREDIBLE_PCT), n)
    return table


def _fit_logit(X: pd.DataFrame, y: pd.Series):
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    if len(y) < MIN_LOGIT_SAMPLES or y.nunique() < 2:
        return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    model = LogisticRegression(C=LOGIT_C, max_iter=1000)
    model.fit(Xs, y.values)
    return (scaler, model)


def fit_window(train_df: pd.DataFrame, horizon: int = 5, cost_threshold: float = 0.0008,
               min_samples: int = MIN_STATE_SAMPLES, feature_cols: list = None) -> ModelBundle:
    feature_cols = feature_cols or FEATURE_COLUMNS
    labels = make_labels(train_df, horizon, cost_threshold)
    states = train_df["direction_state"]

    return ModelBundle(
        markov_long=_fit_markov(states, labels["y_long"], min_samples),
        markov_short=_fit_markov(states, labels["y_short"], min_samples),
        logit_long=_fit_logit(train_df[feature_cols], labels["y_long"]),
        logit_short=_fit_logit(train_df[feature_cols], labels["y_short"]),
        feature_cols=feature_cols,
        min_samples=min_samples,
        base_rate_long=float(labels["y_long"].mean(skipna=True)),
        base_rate_short=float(labels["y_short"].mean(skipna=True)),
    )


def predict(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Per-bar predicted probabilities from both layers, aligned to df's
    index. NaN wherever the state was never seen (enough) in training or a
    logistic layer couldn't be fit -- callers must treat NaN as "no trade",
    never coerce to a neutral probability."""
    out = pd.DataFrame(index=df.index)

    def markov_cols(table):
        mapped = df["direction_state"].map(table)
        p = mapped.apply(lambda t: t[0] if isinstance(t, tuple) else np.nan)
        n = mapped.apply(lambda t: t[1] if isinstance(t, tuple) else 0)
        return p.values, n.values

    out["p_long_markov"], out["n_long_markov"] = markov_cols(bundle.markov_long)
    out["p_short_markov"], out["n_short_markov"] = markov_cols(bundle.markov_short)
    out["base_rate_long"] = bundle.base_rate_long
    out["base_rate_short"] = bundle.base_rate_short

    for side, logit in (("long", bundle.logit_long), ("short", bundle.logit_short)):
        col = f"p_{side}_logit"
        if logit is None:
            out[col] = np.nan
            continue
        scaler, model = logit
        X = df[bundle.feature_cols]
        mask = X.notna().all(axis=1)
        proba = np.full(len(df), np.nan)
        if mask.any():
            proba[mask.values] = model.predict_proba(scaler.transform(X[mask].values))[:, 1]
        out[col] = proba

    return out


def walk_forward_predict(df: pd.DataFrame, train_bars: int, test_bars: int, **fit_kwargs) -> pd.DataFrame:
    """Roll a trailing `train_bars` fit window forward by `test_bars` at a
    time; every fold's predictions come from a model fit only on bars
    strictly before that fold. Returns predictions aligned to df's index,
    NaN before the first fold (warmup)."""
    n = len(df)
    pred_frames = []
    start = train_bars
    while start + test_bars <= n:
        train_slice = df.iloc[start - train_bars:start]
        test_slice = df.iloc[start:start + test_bars]
        bundle = fit_window(train_slice, **fit_kwargs)
        pred_frames.append(predict(test_slice, bundle))
        start += test_bars

    if not pred_frames:
        raise ValueError(
            f"Not enough bars ({n}) for even one walk-forward fold "
            f"(train_bars={train_bars}, test_bars={test_bars})"
        )
    return pd.concat(pred_frames).reindex(df.index)
