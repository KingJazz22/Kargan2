"""Event-driven backtester for the channel-fade + MACD/ADX strategy (see
strategy_channel_macd.py).

Same lookahead-safe convention as this repo's other engines: an entry signal
decided on a bar's close executes at the next bar's open.

Exit is a two-leg take-profit ladder with a stop-loss:

  - SL at entry -/+ sl_atr_mult * ATR(14 @ entry).
  - TP1 at entry -/+ tp1_atr_mult * ATR (default 2.0x) -- closes tp1_pct of
    the position (e.g. 0.5 = half).
  - Once TP1 fills, SL moves to breakeven (entry_price) for the remainder.
  - TP2 closes whatever remains after TP1. Two modes, via tp2_mode:
      "atr"         -- fixed level, entry -/+ tp2_atr_mult * ATR (the
                        original design from this round).
      "opposite_ma" -- price touching the OPPOSITE channel band (max_ma for
                        longs, min_ma for shorts), re-checked live every bar
                        since it's a moving average, not a fixed price. This
                        is the original mean-reversion target from before
                        the SL/TP ladder existed -- see FINDINGS_CHANNEL_
                        MACD.md for why it was retired as the SOLE exit (a
                        losing trade could get "closed at target" simply
                        because the band drifted down to meet a falling
                        price). Now that a real stop-loss + breakeven exist
                        upstream of it, this mode re-tests that same target
                        with the downside capped this time.
    No runner is left after TP2 either way; the position is always fully
    closed by either the (breakeven) stop or TP2.
  - Stop-loss/breakeven-stop is checked before that leg's take-profit on any
    bar where both could fire (conservative tie-break, same convention as
    backtest_pcse_final.py). If TP1 fills AND TP2 is also reached within the
    SAME bar, TP2 is allowed to fire immediately after TP1 in that same bar
    rather than waiting for the next bar -- price legitimately reached both
    levels within the bar's range.
  - Fill prices are gap-aware for take-profits (can't claim a better-than-
    market price on a favorable gap) and exact for stops (no gap adjustment).

Each partial fill (TP1 leg, and the final leg) is recorded as its own Trade
row, sharing entry_date/entry_price/side but with its own shares/risk_amount
(split proportionally: risk_amount*tp1_pct for the TP1 leg, the remainder
for the final leg) and its own exit_date/exit_price/exit_reason -- this lets
basket.py's existing pooled-R-multiple stats aggregate a partially-exited
trade correctly without any change to basket.py itself.

Position sizing is real risk-based sizing again (now that there's a real
stop): risk_amount = cash*risk_pct, shares = risk_amount / (sl_atr_mult *
atr_at_entry), capped by available cash.

At most ONE position per symbol at a time: entries are only queued when
`pos is None and pending_entry is None`, so a new signal while a trade (or
partial remainder of one) is open is simply ignored, never stacked.

No intraday session-time restriction -- entries are evaluated on every bar
regardless of time of day. Works on any bar construction: normal time bars
at any granularity (1m, 2m, ...) or Renko bricks (see renko.py) -- the
strategy and backtester only care that the input has Open/High/Low/Close
columns, not that bars are evenly spaced in time.
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
    tp1_atr_mult: float = 2.0,
    tp1_pct: float = 0.5,
    tp2_atr_mult: float = 3.0,
    tp2_mode: str = "atr",
    cost_per_trade_pct: float = 0.0005,
):
    if tp2_mode not in ("atr", "opposite_ma"):
        raise ValueError(f"tp2_mode must be 'atr' or 'opposite_ma', got {tp2_mode!r}")

    prep = strat.prepare(df, **(strategy_params or {}))
    n = len(prep)
    if n <= WARMUP_BARS + 5:
        raise ValueError(f"Not enough bars ({n}) for warmup ({WARMUP_BARS})")

    cash = starting_equity
    pos = None  # dict: open-position working state (may close in 1 or 2 legs)
    pending_entry = None
    trades: list[Trade] = []
    equity_curve = []

    # itertuples (not iterrows) -- iterrows builds a full Series per row,
    # which dominates runtime on 1-minute bars (100k-400k+ rows/symbol).
    cols = ["Open", "High", "Low", "Close", "atr", "min_ma", "max_ma", "long_entry", "short_entry"]
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
                        tp1_price = entry_price + tp1_atr_mult * atr_at_signal
                        tp2_price = entry_price + tp2_atr_mult * atr_at_signal
                    else:
                        stop_price = entry_price + stop_distance
                        tp1_price = entry_price - tp1_atr_mult * atr_at_signal
                        tp2_price = entry_price - tp2_atr_mult * atr_at_signal
                    pos = {
                        "side": side, "entry_date": date, "entry_price": entry_price,
                        "shares": shares, "remaining_shares": shares,
                        "atr_at_entry": atr_at_signal, "risk_amount": risk_amount,
                        "stop_price": stop_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
                        "tp1_filled": False,
                    }
                    cash += -shares * entry_price if side == "long" else shares * entry_price
            pending_entry = None

        # 2. manage open position: SL/breakeven-SL first (conservative), then
        #    the next TP leg -- both intrabar resting orders
        if pos is not None:
            long_side = pos["side"] == "long"

            def make_trade(shares, fill_price, reason):
                exit_price = fill_price * (1 - cost_per_trade_pct) if long_side else fill_price * (1 + cost_per_trade_pct)
                leg_risk = pos["risk_amount"] * tp1_pct if reason == "tp1" else pos["risk_amount"] * (1 - tp1_pct)
                return Trade(
                    side=pos["side"], entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                    shares=shares, atr_at_entry=pos["atr_at_entry"], risk_amount=leg_risk,
                    exit_date=date, exit_price=exit_price, exit_reason=reason,
                ), exit_price

            # TP2 level: fixed ATR distance set at entry, or the OPPOSITE
            # channel band re-read live every bar (it's a moving average).
            if tp2_mode == "atr":
                tp2_level = pos["tp2_price"]
            else:
                tp2_level = row.max_ma if long_side else row.min_ma
            tp2_valid = pd.notna(tp2_level)

            if not pos["tp1_filled"]:
                stop_hit = row.Low <= pos["stop_price"] if long_side else row.High >= pos["stop_price"]
                if stop_hit:
                    stop_fill = row.Open if (long_side and row.Open <= pos["stop_price"]) or (not long_side and row.Open >= pos["stop_price"]) else pos["stop_price"]
                    trade, exit_price = make_trade(pos["remaining_shares"], stop_fill, "stop_loss")
                    cash += pos["remaining_shares"] * exit_price if long_side else -pos["remaining_shares"] * exit_price
                    trades.append(trade)
                    pos = None
                else:
                    tp1_hit = row.High >= pos["tp1_price"] if long_side else row.Low <= pos["tp1_price"]
                    if tp1_hit:
                        fill = row.Open if (long_side and row.Open >= pos["tp1_price"]) or (not long_side and row.Open <= pos["tp1_price"]) else pos["tp1_price"]
                        leg_shares = pos["shares"] * tp1_pct
                        trade, exit_price = make_trade(leg_shares, fill, "tp1")
                        cash += leg_shares * exit_price if long_side else -leg_shares * exit_price
                        trades.append(trade)
                        pos["remaining_shares"] -= leg_shares
                        pos["tp1_filled"] = True
                        pos["stop_price"] = pos["entry_price"]  # breakeven

                        tp2_hit_same_bar = tp2_valid and (row.High >= tp2_level if long_side else row.Low <= tp2_level)
                        if tp2_hit_same_bar:
                            fill2 = row.Open if (long_side and row.Open >= tp2_level) or (not long_side and row.Open <= tp2_level) else tp2_level
                            trade2, exit_price2 = make_trade(pos["remaining_shares"], fill2, "tp2")
                            cash += pos["remaining_shares"] * exit_price2 if long_side else -pos["remaining_shares"] * exit_price2
                            trades.append(trade2)
                            pos = None
            else:
                stop_hit = row.Low <= pos["stop_price"] if long_side else row.High >= pos["stop_price"]
                if stop_hit:
                    stop_fill = row.Open if (long_side and row.Open <= pos["stop_price"]) or (not long_side and row.Open >= pos["stop_price"]) else pos["stop_price"]
                    trade, exit_price = make_trade(pos["remaining_shares"], stop_fill, "breakeven_stop")
                    cash += pos["remaining_shares"] * exit_price if long_side else -pos["remaining_shares"] * exit_price
                    trades.append(trade)
                    pos = None
                else:
                    tp2_hit = tp2_valid and (row.High >= tp2_level if long_side else row.Low <= tp2_level)
                    if tp2_hit:
                        fill = row.Open if (long_side and row.Open >= tp2_level) or (not long_side and row.Open <= tp2_level) else tp2_level
                        trade, exit_price = make_trade(pos["remaining_shares"], fill, "tp2")
                        cash += pos["remaining_shares"] * exit_price if long_side else -pos["remaining_shares"] * exit_price
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
            equity = cash + pos["remaining_shares"] * row.Close if pos["side"] == "long" else cash - pos["remaining_shares"] * row.Close
        else:
            equity = cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df
