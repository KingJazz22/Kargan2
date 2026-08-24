"""Event-driven backtester for PCSE's final validated configuration (see
strategy_pcse_final.py, FINDINGS_CANDLE_PROB.md). Same lookahead-safe
convention as this repo's other engines: the entry signal is decided on a
bar's close and executes at the next bar's open; the stop-loss and take-
profit are both resting orders that can fill INTRABAR the moment price
touches them. If both would trigger on the same bar, the stop-loss takes
priority (conservative).

Much simpler than the original strategy_candle_prob/backtest_candle_prob
engine on purpose: no trailing stop, no ratchet, no probability-decay exit --
the whole point of the exit-design search this file's config came out of was
that those mechanisms were destroying the entries' edge, not helping it.

Short is the mirror image of long throughout (SL above entry, TP at the
Bollinger LOWER band) and must be enabled explicitly via
`strategy_params={"long_only": False}` -- it has NOT been independently
validated the way the long side has (see FINDINGS_CANDLE_PROB.md's short-side
retest), consistent with this project's long-standing rule of never assuming
a long-side result transfers to short.

`sl_mult`/`bb_num_std` are exposed as run_backtest arguments (not just module
constants) specifically so exit-parameter sweeps (across sides, timeframes)
can override them without monkeypatching strategy_pcse_final.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import strategy_pcse_final as strat

WARMUP_BARS = strat.entry_strat.TRAIN_BARS + 5


@dataclass
class Trade:
    side: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop_distance: float          # "1R" distance, fixed at entry
    risk_amount: float
    exit_date: pd.Timestamp = None
    exit_price: float = None
    exit_reason: str = None

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


def isolate_side(prep: pd.DataFrame, side: str) -> pd.DataFrame:
    """Return a copy of a (long_only=False) prepared DataFrame with the
    OTHER side's entry forced off -- lets long and short be backtested
    independently (no competition for the single open-position slot) without
    re-running the expensive walk-forward fit twice."""
    out = prep.copy()
    if side == "long":
        out["short_entry"] = False
    else:
        out["long_entry"] = False
    return out


def run_backtest(df: pd.DataFrame, starting_equity: float = 100_000.0, risk_pct: float = 0.01,
                  strategy_params: dict = None, cost_per_trade_pct: float = 0.0005,
                  sl_mult: float = None, bb_num_std: float = None):
    prep = strat.prepare(df, **(strategy_params or {}))
    n = len(prep)
    if n <= WARMUP_BARS + 5:
        raise ValueError(f"Not enough bars ({n}) for warmup ({WARMUP_BARS})")
    rows = prep.iloc[WARMUP_BARS:]
    return simulate_rows(rows, starting_equity, risk_pct, cost_per_trade_pct, sl_mult, bb_num_std)


def simulate_rows(rows: pd.DataFrame, starting_equity: float = 100_000.0, risk_pct: float = 0.01,
                   cost_per_trade_pct: float = 0.0005, sl_mult: float = None, bb_num_std: float = None):
    sl_mult = strat.SL_MULT if sl_mult is None else sl_mult
    bb_num_std = strat.BB_NUM_STD if bb_num_std is None else bb_num_std

    cash = starting_equity
    position: Trade = None
    pending_entry = None
    trades: list[Trade] = []
    equity_curve = []

    for date, row in rows.iterrows():
        # 1. execute pending entry at this bar's open
        if position is None and pending_entry is not None:
            side, ret_stdev_at_signal = pending_entry
            raw_price = row["Open"]
            entry_price = raw_price * (1 + cost_per_trade_pct) if side == "long" else raw_price * (1 - cost_per_trade_pct)
            stop_distance = sl_mult * ret_stdev_at_signal * entry_price

            if stop_distance > 0 and not np.isnan(stop_distance):
                risk_amount = cash * risk_pct
                shares = min(risk_amount / stop_distance, cash / entry_price)
                if shares > 0:
                    position = Trade(
                        side=side, entry_date=date, entry_price=entry_price, shares=shares,
                        stop_distance=stop_distance, risk_amount=risk_amount,
                    )
                    cash += -shares * entry_price if side == "long" else shares * entry_price
            pending_entry = None

        # 2. manage open position: SL first (conservative), then TP touch -- both intrabar
        if position is not None:
            if position.side == "long":
                sl_price = position.entry_price - position.stop_distance
                tp_price = None
                if pd.notna(row["bb_mid"]) and pd.notna(row["bb_std"]):
                    candidate = row["bb_mid"] + bb_num_std * row["bb_std"]
                    if candidate > position.entry_price:
                        tp_price = candidate

                fill_price, reason = None, None
                if row["Low"] <= sl_price:
                    fill_price = row["Open"] if row["Open"] <= sl_price else sl_price
                    reason = "stop_loss"
                elif tp_price is not None and row["High"] >= tp_price:
                    fill_price = row["Open"] if row["Open"] >= tp_price else tp_price
                    reason = "take_profit"
            else:
                sl_price = position.entry_price + position.stop_distance
                tp_price = None
                if pd.notna(row["bb_mid"]) and pd.notna(row["bb_std"]):
                    candidate = row["bb_mid"] - bb_num_std * row["bb_std"]
                    if candidate < position.entry_price:
                        tp_price = candidate

                fill_price, reason = None, None
                if row["High"] >= sl_price:
                    fill_price = row["Open"] if row["Open"] >= sl_price else sl_price
                    reason = "stop_loss"
                elif tp_price is not None and row["Low"] <= tp_price:
                    fill_price = row["Open"] if row["Open"] <= tp_price else tp_price
                    reason = "take_profit"

            if fill_price is not None:
                exit_price = fill_price * (1 - cost_per_trade_pct) if position.side == "long" else fill_price * (1 + cost_per_trade_pct)
                position.exit_date, position.exit_price, position.exit_reason = date, exit_price, reason
                cash += position.shares * exit_price if position.side == "long" else -position.shares * exit_price
                trades.append(position)
                position = None

        # 3. entry signal -> queue for next bar's open
        if position is None and pending_entry is None:
            if bool(row["long_entry"]):
                pending_entry = ("long", row["ret_stdev"])
            elif bool(row.get("short_entry", False)):
                pending_entry = ("short", row["ret_stdev"])

        # 4. mark-to-market equity
        if position is not None:
            equity = cash + position.shares * row["Close"] if position.side == "long" else cash - position.shares * row["Close"]
        else:
            equity = cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df
