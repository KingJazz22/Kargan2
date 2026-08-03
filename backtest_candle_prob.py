"""Event-driven backtester for the Probabilistic Candle-State Edge (PCSE)
strategy. Same lookahead-safe convention as this repo's other engines:
signals decided on a bar's close execute at the next bar's open. The stop is
two-stage and confidence-adaptive:

  1. A hard initial stop (HARD_STOP_MULT x entry-time return-stdev) protects
     against a fast reversal before the trail has anything to anchor to --
     the same two-stage shape as Breakout Hunter's initial_stop/emergency_stop,
     but sized off the trade's own realized-return volatility rather than ATR.
  2. Once profit clears TRAIL_ACTIVATE_MULT x that same stdev, a trailing
     stop engages, sized wider or tighter (TRAIL_MULT_LOW..HIGH) by how
     confident the entry signal was, and ratcheted tighter at 1R/2R/3R
     milestones to lock in gains.
  3. Independently of both, if the model's own confidence in the held side
     decays back toward 50/50, the position is closed early ("edge_decay").

Also implements the random-entry permutation test: the standard guard
against candle-pattern-mining false discovery. It replays the *exact same*
exit/sizing engine on randomly chosen entry bars (same count and side mix as
the real signals, same underlying confidence/volatility columns) many times,
and checks whether the real signal's pooled t-stat actually beats picking
entries at random -- entry *timing* is the only thing being tested here; the
walk-forward split in model_candle_prob.py is what guards against the model
itself being overfit to one window.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import strategy_candle_prob as strat

WARMUP_BARS = strat.TRAIN_BARS + 5


@dataclass
class Trade:
    side: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    confidence: float
    stop_distance: float          # "1R" distance, fixed at entry
    risk_amount: float
    exit_date: pd.Timestamp = None
    exit_price: float = None
    exit_reason: str = None
    trail_active: bool = False
    trail_stop: float = None
    extreme: float = None
    bars_held: int = 0

    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        diff = self.exit_price - self.entry_price
        if self.side == "short":
            diff = -diff
        return diff * self.shares

    def r_multiple(self) -> float:
        if self.risk_amount == 0:
            return 0.0
        return self.pnl() / self.risk_amount


def _ratchet_cap(position: Trade, current_r: float) -> float:
    cap = float("inf")
    for level, mult in zip(strat.RATCHET_LEVELS, strat.RATCHET_TRAIL_CAPS):
        if current_r >= level:
            cap = min(cap, mult * position.stop_distance)
    return cap


def _entry_from_signals(i: int, row: pd.Series):
    if bool(row["long_entry"]):
        return ("long", row["long_confidence"], row["ret_stdev"])
    if bool(row["short_entry"]):
        return ("short", row["short_confidence"], row["ret_stdev"])
    return None


def _simulate(rows: pd.DataFrame, pick_entry, starting_equity: float, risk_pct: float,
              cost_per_trade_pct: float, horizon: int) -> tuple[list, pd.DataFrame]:
    cash = starting_equity
    position: Trade = None
    pending_entry = None
    pending_exit_reason = None
    trades: list[Trade] = []
    equity_curve = []

    for i, (date, row) in enumerate(rows.iterrows()):
        # 1. execute pending exit at today's open (with a slippage/cost haircut)
        if position is not None and pending_exit_reason is not None:
            raw_price = row["Open"]
            exit_price = raw_price * (1 - cost_per_trade_pct) if position.side == "long" else raw_price * (1 + cost_per_trade_pct)
            position.exit_date, position.exit_price, position.exit_reason = date, exit_price, pending_exit_reason
            cash += position.shares * exit_price if position.side == "long" else -position.shares * exit_price
            trades.append(position)
            position, pending_exit_reason = None, None

        # 2. execute pending entry at today's open
        if position is None and pending_entry is not None:
            side, confidence, ret_stdev_at_signal = pending_entry
            raw_price = row["Open"]
            entry_price = raw_price * (1 + cost_per_trade_pct) if side == "long" else raw_price * (1 - cost_per_trade_pct)
            dollar_stdev = entry_price * ret_stdev_at_signal
            stop_distance = strat.HARD_STOP_MULT * dollar_stdev

            if stop_distance > 0 and not np.isnan(stop_distance):
                risk_amount = cash * risk_pct
                shares = min(risk_amount / stop_distance, cash / entry_price)
                if shares > 0:
                    position = Trade(
                        side=side, entry_date=date, entry_price=entry_price, shares=shares,
                        confidence=confidence, stop_distance=stop_distance, risk_amount=risk_amount,
                        extreme=entry_price,
                    )
                    if side == "long":
                        position.trail_stop = entry_price - stop_distance
                        cash -= shares * entry_price
                    else:
                        position.trail_stop = entry_price + stop_distance
                        cash += shares * entry_price
            pending_entry = None

        # 3. manage open position: activate/advance trail, ratchet, check intrabar stop fill
        if position is not None:
            position.bars_held += 1
            dollar_stdev_now = row["Close"] * row["ret_stdev"]
            trail_mult = strat.confidence_to_trail_mult(position.confidence)

            if position.side == "long":
                position.extreme = max(position.extreme, row["Close"])
                profit = position.extreme - position.entry_price
                if not position.trail_active and profit >= strat.TRAIL_ACTIVATE_MULT * dollar_stdev_now:
                    position.trail_active = True
                if position.trail_active:
                    current_r = profit / position.stop_distance
                    trail_distance = min(trail_mult * dollar_stdev_now, _ratchet_cap(position, current_r))
                    position.trail_stop = max(position.trail_stop, position.extreme - trail_distance)
            else:
                position.extreme = min(position.extreme, row["Close"])
                profit = position.entry_price - position.extreme
                if not position.trail_active and profit >= strat.TRAIL_ACTIVATE_MULT * dollar_stdev_now:
                    position.trail_active = True
                if position.trail_active:
                    current_r = profit / position.stop_distance
                    trail_distance = min(trail_mult * dollar_stdev_now, _ratchet_cap(position, current_r))
                    position.trail_stop = min(position.trail_stop, position.extreme + trail_distance)

            fill_price = None
            if position.side == "long" and row["Low"] <= position.trail_stop:
                fill_price = position.trail_stop
            elif position.side == "short" and row["High"] >= position.trail_stop:
                fill_price = position.trail_stop

            if fill_price is not None:
                position.exit_date, position.exit_price = date, fill_price
                position.exit_reason = "trailing_stop" if position.trail_active else "hard_stop"
                cash += position.shares * fill_price if position.side == "long" else -position.shares * fill_price
                trades.append(position)
                position = None

        # 4. probability-decay early exit -> queue for next bar's open. Gated on
        # the model's own prediction horizon having elapsed: direction_state (and
        # therefore *_confidence) is recomputed fresh every bar from that bar's own
        # candle, so checking it against the ORIGINAL trade's thesis is only
        # meaningful once that thesis has had time to play out -- checking it
        # immediately just re-asks "does today's unrelated candle also look
        # long-worthy", which caused near-universal 1-bar exits before this gate.
        if position is not None and position.bars_held >= horizon:
            live_conf = row["long_confidence"] if position.side == "long" else row["short_confidence"]
            if pd.notna(live_conf) and live_conf < strat.EDGE_MARGIN * strat.DECAY_HALF_LIFE_FRAC:
                pending_exit_reason = "edge_decay"

        # 5. entry signals -> queue for next bar's open
        if position is None and pending_entry is None:
            pending_entry = pick_entry(i, row)

        # 6. mark-to-market equity
        if position is not None:
            equity = cash + position.shares * row["Close"] if position.side == "long" else cash - position.shares * row["Close"]
        else:
            equity = cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df


def _prep_rows(df: pd.DataFrame, strategy_params: dict) -> tuple[pd.DataFrame, int]:
    strategy_params = strategy_params or {}
    horizon = strategy_params.get("horizon", strat.HORIZON)
    prep = strat.prepare(df, **strategy_params)
    n = len(prep)
    if n <= WARMUP_BARS + 5:
        raise ValueError(f"Not enough bars ({n}) for warmup ({WARMUP_BARS})")
    return prep.iloc[WARMUP_BARS:], horizon


def run_backtest(df: pd.DataFrame, starting_equity: float = 100_000.0, risk_pct: float = 0.01,
                  strategy_params: dict = None, cost_per_trade_pct: float = 0.0005):
    rows, horizon = _prep_rows(df, strategy_params)
    return _simulate(rows, _entry_from_signals, starting_equity, risk_pct, cost_per_trade_pct, horizon)


def _pooled_stats(trades: list) -> dict:
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return {"n": 0, "avg_r": 0.0, "t_stat": 0.0}
    r = np.array([t.r_multiple() for t in closed])
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    t_stat = float(r.mean() / se) if se > 0 else 0.0
    return {"n": len(r), "avg_r": float(r.mean()), "t_stat": t_stat}


def random_entry_permutation_test(df: pd.DataFrame, n_shuffles: int = 200, seed: int = 0,
                                   strategy_params: dict = None, starting_equity: float = 100_000.0,
                                   risk_pct: float = 0.01, cost_per_trade_pct: float = 0.0005) -> dict:
    """Fit/predict once (expensive), then hold the model's confidence and
    volatility columns fixed and re-run only the entry-timing + exit engine
    with randomly chosen entry bars, n_shuffles times, per side. Reports what
    percentile the REAL signal's pooled t-stat falls at within that null
    distribution -- >=95 is the bar for "the timing itself has real edge,
    not just noise that looked good once."""
    rows, horizon = _prep_rows(df, strategy_params)

    real_trades, _ = _simulate(rows, _entry_from_signals, starting_equity, risk_pct, cost_per_trade_pct, horizon)
    real_long = [t for t in real_trades if t.side == "long"]
    real_short = [t for t in real_trades if t.side == "short"]
    real_stats = {"long": _pooled_stats(real_long), "short": _pooled_stats(real_short)}

    n_bars = len(rows)
    n_long, n_short = len(real_long), len(real_short)
    rng = np.random.default_rng(seed)

    null_t_long, null_t_short = [], []
    for _ in range(n_shuffles):
        trial_trades = _random_trial_trades(rows, n_long, n_short, horizon, rng, starting_equity, risk_pct, cost_per_trade_pct)
        if n_long:
            null_t_long.append(_pooled_stats([t for t in trial_trades if t.side == "long"])["t_stat"])
        if n_short:
            null_t_short.append(_pooled_stats([t for t in trial_trades if t.side == "short"])["t_stat"])

    def percentile_of(real_t, null_arr):
        if not null_arr:
            return None
        return float((np.array(null_arr) < real_t).mean() * 100)

    return {
        "real": real_stats,
        "null_percentile_long": percentile_of(real_stats["long"]["t_stat"], null_t_long),
        "null_percentile_short": percentile_of(real_stats["short"]["t_stat"], null_t_short),
        "n_shuffles": n_shuffles,
    }


def _random_trial_trades(rows: pd.DataFrame, n_long: int, n_short: int, horizon: int, rng: np.random.Generator,
                          starting_equity: float, risk_pct: float, cost_per_trade_pct: float) -> list:
    n_bars = len(rows)
    chosen = rng.choice(n_bars, size=min(n_long + n_short, n_bars), replace=False)
    long_idx = set(chosen[:n_long].tolist())
    short_idx = set(chosen[n_long:n_long + n_short].tolist())

    def pick_random(i, row, long_idx=long_idx, short_idx=short_idx):
        if i in long_idx:
            return ("long", strat.EDGE_MARGIN, row["ret_stdev"])
        if i in short_idx:
            return ("short", strat.EDGE_MARGIN, row["ret_stdev"])
        return None

    trades, _ = _simulate(rows, pick_random, starting_equity, risk_pct, cost_per_trade_pct, horizon)
    return trades


def pooled_permutation_test(dfs: dict, n_shuffles: int = 200, seed: int = 0, strategy_params: dict = None,
                             starting_equity: float = 100_000.0, risk_pct: float = 0.01,
                             cost_per_trade_pct: float = 0.0005, verbose: bool = True) -> dict:
    """Basket-wide version of the random-entry permutation test: fits/predicts
    once per symbol, then for each of n_shuffles trials draws random entries
    independently per symbol (matching that symbol's own real entry counts)
    and pools the resulting trades across the WHOLE basket before computing
    the null t-stat -- mirroring exactly how the real pooled result
    (basket.run_basket) is computed, so the comparison is apples-to-apples."""
    per_symbol_rows = {}
    real_trades_all = []
    skipped = []
    for symbol, df in dfs.items():
        try:
            rows, horizon = _prep_rows(df, strategy_params)
        except Exception as e:
            skipped.append((symbol, str(e)))
            continue
        real_trades, _ = _simulate(rows, _entry_from_signals, starting_equity, risk_pct, cost_per_trade_pct, horizon)
        per_symbol_rows[symbol] = (rows, horizon, real_trades)
        real_trades_all.extend(real_trades)
        if verbose:
            print(f"  prepped {symbol}: bars={len(rows)} real_trades={len([t for t in real_trades if t.exit_price is not None])}")

    real_long = [t for t in real_trades_all if t.side == "long"]
    real_short = [t for t in real_trades_all if t.side == "short"]
    real_stats = {"long": _pooled_stats(real_long), "short": _pooled_stats(real_short)}

    rng = np.random.default_rng(seed)
    null_t_long, null_t_short = [], []
    for shuffle_i in range(n_shuffles):
        trial_long, trial_short = [], []
        for symbol, (rows, horizon, real_trades) in per_symbol_rows.items():
            n_long = len([t for t in real_trades if t.side == "long"])
            n_short = len([t for t in real_trades if t.side == "short"])
            if n_long == 0 and n_short == 0:
                continue
            trial_trades = _random_trial_trades(rows, n_long, n_short, horizon, rng, starting_equity, risk_pct, cost_per_trade_pct)
            trial_long.extend([t for t in trial_trades if t.side == "long"])
            trial_short.extend([t for t in trial_trades if t.side == "short"])
        if real_long:
            null_t_long.append(_pooled_stats(trial_long)["t_stat"])
        if real_short:
            null_t_short.append(_pooled_stats(trial_short)["t_stat"])
        if verbose and (shuffle_i + 1) % 50 == 0:
            print(f"  permutation trial {shuffle_i + 1}/{n_shuffles}")

    def percentile_of(real_t, null_arr):
        if not null_arr:
            return None
        return float((np.array(null_arr) < real_t).mean() * 100)

    return {
        "real": real_stats,
        "null_percentile_long": percentile_of(real_stats["long"]["t_stat"], null_t_long),
        "null_percentile_short": percentile_of(real_stats["short"]["t_stat"], null_t_short),
        "null_t_long": null_t_long,
        "null_t_short": null_t_short,
        "n_shuffles": n_shuffles,
        "skipped": skipped,
    }
