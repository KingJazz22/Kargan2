"""Event-driven backtester for the structure-confirmed breakout strategy.

Long only. Same lookahead-safe convention as the other engines: signal on
close, execute at next bar's open; stops are resting orders, fill intrabar.
Two-stage stop, reused unchanged from backtest_breakout.py (Breakout Hunter):
a structural initial stop below the pre-breakout `support` swing low, then an
ATR trailing stop that takes over once the trade has 1x ATR of profit cushion.
"""
from dataclasses import dataclass

import pandas as pd

import strategy_swing_breakout as strat

WARMUP_BARS = 2 * strat.SWING_LEFT + 2 * strat.SWING_RIGHT + 50


@dataclass
class Trade:
    side: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    initial_stop: float
    emergency_stop: float
    atr_at_entry: float
    risk_amount: float
    exit_date: pd.Timestamp = None
    exit_price: float = None
    exit_reason: str = None

    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    def r_multiple(self) -> float:
        if self.risk_amount == 0:
            return 0.0
        return self.pnl() / self.risk_amount


def run_backtest(df: pd.DataFrame, starting_equity: float = 100_000.0, risk_pct: float = 0.01):
    data = strat.prepare(df)
    n = len(data)
    if n <= WARMUP_BARS + 5:
        raise ValueError(f"Not enough bars ({n}) for warmup ({WARMUP_BARS})")

    cash = starting_equity
    position = None
    pending_entry = None
    trades: list[Trade] = []
    equity_curve = []

    rows = data.iloc[WARMUP_BARS:]

    for date, row in rows.iterrows():
        # 1. execute pending entry at today's open
        if position is None and pending_entry is not None:
            support, atr_at_signal = pending_entry
            entry_price = row["Open"]
            initial_stop = support - strat.INITIAL_STOP_ATR_BUFFER * atr_at_signal
            # Anchored beyond initial_stop, not independently from entry -- see
            # FINDINGS_BREAKOUT.md for why an entry-anchored emergency stop can
            # invert and become tighter than the structural stop.
            emergency_stop = initial_stop - strat.EMERGENCY_STOP_BUFFER_ATR * atr_at_signal
            stop_distance = entry_price - initial_stop

            if stop_distance > 0:
                risk_amount = cash * risk_pct
                shares = risk_amount / stop_distance
                shares = min(shares, cash / entry_price)
                if shares > 0:
                    position = Trade(
                        side="long",
                        entry_date=date,
                        entry_price=entry_price,
                        shares=shares,
                        initial_stop=initial_stop,
                        emergency_stop=emergency_stop,
                        atr_at_entry=atr_at_signal,
                        risk_amount=risk_amount,
                    )
                    position.trail_active = False
                    position.trail_stop = None
                    position.extreme = entry_price
                    cash -= shares * entry_price
            pending_entry = None

        # 2. manage open position: update trail, check intrabar stop fills
        if position is not None:
            position.extreme = max(position.extreme, row["High"])
            favorable = row["Close"] - position.entry_price

            if not position.trail_active and favorable >= strat.TRAIL_ACTIVATE_ATR * position.atr_at_entry:
                position.trail_active = True

            if position.trail_active:
                candidate = position.extreme - strat.TRAIL_ATR_MULT * row["atr"]
                position.trail_stop = candidate if position.trail_stop is None else max(position.trail_stop, candidate)

            active_stop = position.trail_stop if position.trail_active else position.initial_stop
            stop_reason = "trailing_stop" if position.trail_active else "initial_stop"

            fill_price = None
            fill_reason = None
            if row["Low"] <= position.emergency_stop:
                fill_price = row["Open"] if row["Open"] <= position.emergency_stop else position.emergency_stop
                fill_reason = "emergency_stop"
            elif row["Low"] <= active_stop:
                fill_price = row["Open"] if row["Open"] <= active_stop else active_stop
                fill_reason = stop_reason

            if fill_price is not None:
                position.exit_date = date
                position.exit_price = fill_price
                position.exit_reason = fill_reason
                cash += position.shares * fill_price
                trades.append(position)
                position = None

        # 3. entry signal -> queue for next bar's open
        if position is None and pending_entry is None:
            if bool(row["long_entry"]):
                pending_entry = (row["support"], row["atr"])

        # 4. mark-to-market equity
        if position is not None:
            equity = cash + position.shares * row["Close"]
        else:
            equity = cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df
