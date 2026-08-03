"""Event-driven backtester for the Adaptive Trend Following strategy.

Signals are decided on a bar's CLOSE and executed at the NEXT bar's OPEN
(no lookahead). Stop-loss / trailing-stop levels are resting price orders
and are allowed to fill intrabar, using that bar's High/Low.

Simplifications (research-grade, not execution-grade):
  - No commissions, slippage, or borrow cost modeled.
  - Shorting is allowed with no margin/borrow constraints.
  - Stops fill exactly at the stop price when touched (no gap-through modeling
    beyond the emergency stop itself being the gap backstop).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import strategy as strat

WARMUP_BARS = strat.EMA_SLOW + 50


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
        diff = self.exit_price - self.entry_price
        if self.side == "short":
            diff = -diff
        return diff * self.shares

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
    pending_exit_reason = None
    trades: list[Trade] = []
    equity_curve = []

    rows = data.iloc[WARMUP_BARS:]

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
            side, swing_extreme, atr_at_signal = pending_entry
            entry_price = row["Open"]
            if side == "long":
                initial_stop = swing_extreme - strat.INITIAL_STOP_ATR_BUFFER * atr_at_signal
                emergency_stop = entry_price - strat.EMERGENCY_ATR_MULT * atr_at_signal
                stop_distance = entry_price - initial_stop
            else:
                initial_stop = swing_extreme + strat.INITIAL_STOP_ATR_BUFFER * atr_at_signal
                emergency_stop = entry_price + strat.EMERGENCY_ATR_MULT * atr_at_signal
                stop_distance = initial_stop - entry_price

            if stop_distance > 0:
                equity_now = cash if position is None else cash
                risk_amount = equity_now * risk_pct
                shares = risk_amount / stop_distance
                shares = min(shares, equity_now / entry_price)  # no leverage
                if shares > 0:
                    position = Trade(
                        side=side,
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
                    if side == "long":
                        cash -= shares * entry_price
                    else:
                        cash += shares * entry_price
            pending_entry = None

        # 3. manage open position: update trail, check intrabar stop fills
        if position is not None:
            if position.side == "long":
                position.extreme = max(position.extreme, row["High"])
                favorable = row["Close"] - position.entry_price
            else:
                position.extreme = min(position.extreme, row["Low"])
                favorable = position.entry_price - row["Close"]

            if not position.trail_active and favorable >= strat.TRAIL_ACTIVATE_ATR * position.atr_at_entry:
                position.trail_active = True

            if position.trail_active:
                if position.side == "long":
                    candidate = position.extreme - strat.TRAIL_ATR_MULT * row["atr"]
                    position.trail_stop = candidate if position.trail_stop is None else max(position.trail_stop, candidate)
                else:
                    candidate = position.extreme + strat.TRAIL_ATR_MULT * row["atr"]
                    position.trail_stop = candidate if position.trail_stop is None else min(position.trail_stop, candidate)

            active_stop = position.trail_stop if position.trail_active else position.initial_stop
            stop_reason = "trailing_stop" if position.trail_active else "initial_stop"

            fill_price = None
            fill_reason = None
            if position.side == "long":
                if row["Low"] <= position.emergency_stop:
                    fill_price, fill_reason = position.emergency_stop, "emergency_stop"
                elif row["Low"] <= active_stop:
                    fill_price, fill_reason = active_stop, stop_reason
            else:
                if row["High"] >= position.emergency_stop:
                    fill_price, fill_reason = position.emergency_stop, "emergency_stop"
                elif row["High"] >= active_stop:
                    fill_price, fill_reason = active_stop, stop_reason

            if fill_price is not None:
                position.exit_date = date
                position.exit_price = fill_price
                position.exit_reason = fill_reason
                if position.side == "long":
                    cash += position.shares * fill_price
                else:
                    cash -= position.shares * fill_price
                trades.append(position)
                position = None

        # 4. close-based exit signals -> queue for next bar's open
        if position is not None:
            if bool(row["hard_kill"]):
                pending_exit_reason = "adx_kill"
            elif position.side == "long" and bool(row["long_reversal_exit"]):
                pending_exit_reason = "trend_reversal"
            elif position.side == "short" and bool(row["short_reversal_exit"]):
                pending_exit_reason = "trend_reversal"
            elif position.side == "long" and bool(row["long_backstop_exit"]):
                pending_exit_reason = "ema_cross_backstop"
            elif position.side == "short" and bool(row["short_backstop_exit"]):
                pending_exit_reason = "ema_cross_backstop"

        # 5. entry signals -> queue for next bar's open (only if flat and nothing pending)
        if position is None and pending_entry is None:
            lookback = data.loc[:date].tail(3)
            if bool(row["long_entry"]):
                swing_low = lookback["Low"].min()
                pending_entry = ("long", swing_low, row["atr"])
            elif bool(row["short_entry"]):
                swing_high = lookback["High"].max()
                pending_entry = ("short", swing_high, row["atr"])

        # 6. mark-to-market equity
        if position is not None:
            equity = cash + position.shares * row["Close"] if position.side == "long" else cash - position.shares * row["Close"]
        else:
            equity = cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df


def summarize(trades: list[Trade], equity_df: pd.DataFrame, starting_equity: float) -> dict:
    closed = [t for t in trades if t.exit_price is not None]
    n = len(closed)
    if n == 0:
        return {"num_trades": 0}

    pnls = np.array([t.pnl() for t in closed])
    r_mults = np.array([t.r_multiple() for t in closed])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    final_equity = equity_df["equity"].iloc[-1]
    total_return_pct = (final_equity / starting_equity - 1) * 100

    daily_returns = equity_df["equity"].pct_change().dropna()
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    running_max = equity_df["equity"].cummax()
    drawdown = equity_df["equity"] / running_max - 1
    max_dd_pct = drawdown.min() * 100

    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr_pct = ((final_equity / starting_equity) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    exit_reasons = pd.Series([t.exit_reason for t in closed]).value_counts().to_dict()

    return {
        "num_trades": n,
        "win_rate_pct": round(100 * len(wins) / n, 1),
        "avg_r_multiple": round(float(r_mults.mean()), 2),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 2) if losses.sum() != 0 else float("inf"),
        "total_return_pct": round(total_return_pct, 1),
        "cagr_pct": round(cagr_pct, 1),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "sharpe": round(float(sharpe), 2),
        "final_equity": round(float(final_equity), 2),
        "exit_reasons": exit_reasons,
    }
