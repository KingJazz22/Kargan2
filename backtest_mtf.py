"""Event-driven backtester for the Multi-Timeframe RSI+Stochastic strategy.

Same lookahead-safe convention as the other backtest engines: signals decided
on an LTF bar's close execute at the next LTF bar's open; the ATR trailing
stop (active from entry, doubling as the stop-loss -- see strategy_mtf.py)
is a resting order that can fill intrabar. `data` is a (ltf_df, htf_df) tuple
so this plugs into basket.run_basket via a fetch_fn that returns the pair.
"""
from dataclasses import dataclass

import pandas as pd

import strategy_mtf as strat

WARMUP_BARS = 100


@dataclass
class Trade:
    side: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    atr_at_entry: float
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


def run_backtest(data, starting_equity: float = 100_000.0, risk_pct: float = 0.01, strategy_params: dict = None):
    ltf_df, htf_df, *rest = data
    trend_filter_df = rest[0] if rest else None
    prep = strat.prepare(ltf_df, htf_df, trend_filter_df=trend_filter_df, **(strategy_params or {}))
    n = len(prep)
    if n <= WARMUP_BARS + 5:
        raise ValueError(f"Not enough bars ({n}) for warmup ({WARMUP_BARS})")

    cash = starting_equity
    position = None
    pending_entry = None
    pending_exit_reason = None
    trades: list[Trade] = []
    equity_curve = []

    rows = prep.iloc[WARMUP_BARS:]

    for date, row in rows.iterrows():
        # 1. execute pending exit at today's open
        if position is not None and pending_exit_reason is not None:
            exit_price = row["Open"]
            position.exit_date = date
            position.exit_price = exit_price
            position.exit_reason = pending_exit_reason
            if position.side == "long":
                cash += position.shares * exit_price
            else:
                cash -= position.shares * exit_price
            trades.append(position)
            position = None
            pending_exit_reason = None

        # 2. execute pending entry at today's open
        if position is None and pending_entry is not None:
            side, atr_at_signal = pending_entry
            entry_price = row["Open"]
            stop_distance = strat.TRAIL_ATR_MULT * atr_at_signal

            if stop_distance > 0:
                risk_amount = cash * risk_pct
                shares = risk_amount / stop_distance
                shares = min(shares, cash / entry_price)
                if shares > 0:
                    position = Trade(
                        side=side,
                        entry_date=date,
                        entry_price=entry_price,
                        shares=shares,
                        atr_at_entry=atr_at_signal,
                        risk_amount=risk_amount,
                    )
                    if side == "long":
                        position.trail_stop = entry_price - stop_distance
                        position.extreme = entry_price
                        cash -= shares * entry_price
                    else:
                        position.trail_stop = entry_price + stop_distance
                        position.extreme = entry_price
                        cash += shares * entry_price
            pending_entry = None

        # 3. manage open position: update trail, check intrabar stop fill
        if position is not None:
            if position.side == "long":
                position.extreme = max(position.extreme, row["Close"])
                candidate = position.extreme - strat.TRAIL_ATR_MULT * row["atr"]
                position.trail_stop = max(position.trail_stop, candidate)
            else:
                position.extreme = min(position.extreme, row["Close"])
                candidate = position.extreme + strat.TRAIL_ATR_MULT * row["atr"]
                position.trail_stop = min(position.trail_stop, candidate)

            fill_price = None
            if position.side == "long" and row["Low"] <= position.trail_stop:
                fill_price = position.trail_stop
            elif position.side == "short" and row["High"] >= position.trail_stop:
                fill_price = position.trail_stop

            if fill_price is not None:
                position.exit_date = date
                position.exit_price = fill_price
                position.exit_reason = "trailing_stop"
                if position.side == "long":
                    cash += position.shares * fill_price
                else:
                    cash -= position.shares * fill_price
                trades.append(position)
                position = None

        # 4. close-based target exit -> queue for next bar's open
        if position is not None:
            if position.side == "long" and bool(row["long_target_exit"]):
                pending_exit_reason = "target_rsi_neutral"
            elif position.side == "short" and bool(row["short_target_exit"]):
                pending_exit_reason = "target_rsi_neutral"

        # 5. entry signals -> queue for next bar's open
        if position is None and pending_entry is None:
            if bool(row["long_entry"]):
                pending_entry = ("long", row["atr"])
            elif bool(row["short_entry"]):
                pending_entry = ("short", row["atr"])

        # 6. mark-to-market equity
        if position is not None:
            equity = cash + position.shares * row["Close"] if position.side == "long" else cash - position.shares * row["Close"]
        else:
            equity = cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df
