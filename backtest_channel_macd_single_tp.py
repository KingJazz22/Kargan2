"""Event-driven backtester for the channel + MACD/ADX family of entry rules
(strategy_channel_macd.py's fade design by default) using the simplest
possible exit: one fixed stop-loss and one fixed take-profit, no partial
legs, no breakeven, no trailing. This is the best-known exit design found
for this family (see FINDINGS_CHANNEL_MACD.md Update 4 -- it beat both the
SL/TP1/TP2 ladder in backtest_channel_macd.py and the two-stage trailing
stop in backtest_channel_macd_trailing.py at every setting compared).

Same lookahead-safe / intrabar-resting-order / gap-aware-TP conventions as
backtest_channel_macd.py; stop-loss checked first on any bar where both
could fire (conservative tie-break).

`strategy_module` lets this same proven exit engine run ANY entry-rule
module that exposes the same `prepare(df, ...) -> DataFrame` contract with
`atr`/`long_entry`/`short_entry` columns -- e.g.
strategy_channel_macd_momentum.py (the breakout-continuation variant) --
without duplicating this file's exit logic per entry-rule experiment.
Defaults to strategy_channel_macd (the original fade design) so existing
callers are unaffected.
"""
from dataclasses import dataclass

import pandas as pd

import strategy_channel_macd as default_strat

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
    tp_atr_mult: float = 3.0,
    cost_per_trade_pct: float = 0.0005,
    strategy_module=None,
    side_filter: str = "both",
):
    if side_filter not in ("both", "long_only", "short_only"):
        raise ValueError(f"side_filter must be 'both', 'long_only', or 'short_only', got {side_filter!r}")

    strat = strategy_module or default_strat
    prep = strat.prepare(df, **(strategy_params or {}))
    if side_filter == "long_only":
        prep = prep.assign(short_entry=False)
    elif side_filter == "short_only":
        prep = prep.assign(long_entry=False)
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
                    if side == "long":
                        stop_price = entry_price - stop_distance
                        tp_price = entry_price + tp_atr_mult * atr_at_signal
                    else:
                        stop_price = entry_price + stop_distance
                        tp_price = entry_price - tp_atr_mult * atr_at_signal
                    pos = {
                        "side": side, "entry_date": date, "entry_price": entry_price,
                        "shares": shares, "atr_at_entry": atr_at_signal, "risk_amount": risk_amount,
                        "stop_price": stop_price, "tp_price": tp_price,
                    }
                    cash += -shares * entry_price if side == "long" else shares * entry_price
            pending_entry = None

        # 2. manage open position: SL first (conservative), then TP -- both intrabar
        if pos is not None:
            long_side = pos["side"] == "long"
            fill_price, reason = None, None

            stop_hit = row.Low <= pos["stop_price"] if long_side else row.High >= pos["stop_price"]
            if stop_hit:
                fill_price = row.Open if (long_side and row.Open <= pos["stop_price"]) or (not long_side and row.Open >= pos["stop_price"]) else pos["stop_price"]
                reason = "stop_loss"
            else:
                tp_hit = row.High >= pos["tp_price"] if long_side else row.Low <= pos["tp_price"]
                if tp_hit:
                    fill_price = row.Open if (long_side and row.Open >= pos["tp_price"]) or (not long_side and row.Open <= pos["tp_price"]) else pos["tp_price"]
                    reason = "take_profit"

            if fill_price is not None:
                exit_price = fill_price * (1 - cost_per_trade_pct) if long_side else fill_price * (1 + cost_per_trade_pct)
                trade = Trade(
                    side=pos["side"], entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                    shares=pos["shares"], atr_at_entry=pos["atr_at_entry"], risk_amount=pos["risk_amount"],
                    exit_date=date, exit_price=exit_price, exit_reason=reason,
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
