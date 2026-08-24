"""Event-driven backtester for the channel-fade + MACD/ADX strategy (see
strategy_channel_macd.py) using a TWO-STAGE STOP instead of the fixed SL/TP1/
TP2 ladder in backtest_channel_macd.py -- same two-stage-stop shape already
proven out in this repo for the Breakout Hunter strategy (backtest_breakout.py):
a fixed initial stop, then once price has moved `trail_activate_atr` * ATR in
the trade's favor, the stop switches to an ATR trailing stop that only ever
tightens toward the current extreme (`ratchets`, never loosens). There is NO
fixed take-profit at all here -- the only way out is a stop (initial or
trailing), so a strong move is free to run as far as it goes before pulling
back trail_atr_mult * ATR from its best point.

Same lookahead-safe convention as the other engines in this repo: entry
signal decided on a bar's close executes at the next bar's open; stops are
resting orders that fill intrabar (exact price, no gap adjustment -- matches
backtest_breakout.py's convention, unlike the gap-aware TP fills in
backtest_channel_macd.py, since there's no take-profit leg here to be
gap-aware about).

Position sizing is the same fixed-fractional risk convention used
throughout this repo: risk_amount = cash*risk_pct, shares = risk_amount /
(sl_atr_mult * atr_at_entry), capped by available cash.

At most ONE position per symbol at a time, same as backtest_channel_macd.py.
"""
from dataclasses import dataclass

import pandas as pd

import strategy_channel_macd as strat

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


def run_backtest(
    df: pd.DataFrame,
    starting_equity: float = 100_000.0,
    risk_pct: float = 0.01,
    strategy_params: dict = None,
    sl_atr_mult: float = 2.0,
    trail_activate_atr: float = 1.0,
    trail_atr_mult: float = 2.0,
    cost_per_trade_pct: float = 0.0005,
):
    prep = strat.prepare(df, **(strategy_params or {}))
    n = len(prep)
    if n <= WARMUP_BARS + 5:
        raise ValueError(f"Not enough bars ({n}) for warmup ({WARMUP_BARS})")

    cash = starting_equity
    pos = None
    pending_entry = None
    trades: list[Trade] = []
    equity_curve = []

    cols = ["Open", "High", "Low", "Close", "atr", "long_entry", "short_entry"]
    rows = prep.iloc[WARMUP_BARS:][cols]

    for row in rows.itertuples(name="Bar"):
        date = row.Index

        # 1. execute pending entry at this bar's open
        if pos is None and pending_entry is not None:
            side, atr_at_signal = pending_entry
            raw_price = row.Open
            entry_price = raw_price * (1 + cost_per_trade_pct) if side == "long" else raw_price * (1 - cost_per_trade_pct)
            stop_distance = sl_atr_mult * atr_at_signal

            if stop_distance > 0 and pd.notna(stop_distance):
                risk_amount = cash * risk_pct
                shares = min(risk_amount / stop_distance, cash / entry_price)
                if shares > 0:
                    initial_stop = entry_price - stop_distance if side == "long" else entry_price + stop_distance
                    pos = {
                        "side": side, "entry_date": date, "entry_price": entry_price,
                        "shares": shares, "atr_at_entry": atr_at_signal, "risk_amount": risk_amount,
                        "initial_stop": initial_stop, "extreme": entry_price,
                        "trail_active": False, "trail_stop": None,
                    }
                    cash += -shares * entry_price if side == "long" else shares * entry_price
            pending_entry = None

        # 2. manage open position: activate/ratchet trail, check intrabar stop fill
        if pos is not None:
            long_side = pos["side"] == "long"

            if long_side:
                pos["extreme"] = max(pos["extreme"], row.High)
                favorable = row.Close - pos["entry_price"]
            else:
                pos["extreme"] = min(pos["extreme"], row.Low)
                favorable = pos["entry_price"] - row.Close

            if not pos["trail_active"] and favorable >= trail_activate_atr * pos["atr_at_entry"]:
                pos["trail_active"] = True

            if pos["trail_active"]:
                if long_side:
                    candidate = pos["extreme"] - trail_atr_mult * row.atr
                    pos["trail_stop"] = candidate if pos["trail_stop"] is None else max(pos["trail_stop"], candidate)
                else:
                    candidate = pos["extreme"] + trail_atr_mult * row.atr
                    pos["trail_stop"] = candidate if pos["trail_stop"] is None else min(pos["trail_stop"], candidate)

            active_stop = pos["trail_stop"] if pos["trail_active"] else pos["initial_stop"]
            stop_reason = "trailing_stop" if pos["trail_active"] else "initial_stop"

            stop_hit = row.Low <= active_stop if long_side else row.High >= active_stop
            if stop_hit:
                stop_fill = row.Open if (long_side and row.Open <= active_stop) or (not long_side and row.Open >= active_stop) else active_stop
                exit_price = stop_fill * (1 - cost_per_trade_pct) if long_side else stop_fill * (1 + cost_per_trade_pct)
                trade = Trade(
                    side=pos["side"], entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                    shares=pos["shares"], atr_at_entry=pos["atr_at_entry"], risk_amount=pos["risk_amount"],
                    exit_date=date, exit_price=exit_price, exit_reason=stop_reason,
                )
                cash += pos["shares"] * exit_price if long_side else -pos["shares"] * exit_price
                trades.append(trade)
                pos = None

        # 3. entry signal -> queue for next bar's open
        if pos is None and pending_entry is None:
            if bool(row.long_entry):
                pending_entry = ("long", row.atr)
            elif bool(row.short_entry):
                pending_entry = ("short", row.atr)

        # 4. mark-to-market equity
        if pos is not None:
            equity = cash + pos["shares"] * row.Close if pos["side"] == "long" else cash - pos["shares"] * row.Close
        else:
            equity = cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df
