"""Combined portfolio backtest: Breakout Hunter (daily) + Multi-Timeframe
RSI+Stochastic long-only (1H+4H) running at the same time, sharing ONE
capital pool -- the two strategies in this project with a demonstrated edge.

Reuses each strategy's already-validated signal/stop logic exactly (both are
purely price/ATR-driven, independent of account size), but replaces each
standalone engine's independent $100k account with one shared ledger: every
new entry (either strategy) sizes off CURRENT total equity (cash + all open
positions marked to market), matching how live_breakout.py already works.

A symbol can have positions open in BOTH strategies simultaneously -- they're
tracked as separate (symbol, strategy) lots with their own entry price and
stop state, same as two independent bots trading one shared brokerage account.

Timing: the master clock is the union of all 1H timestamps (MTF's native
granularity). Breakout Hunter's daily logic -- entries, and its once-per-day
stop check -- fires at the first 1H tick of each new trading day, using the
just-completed prior daily bar (mirrors backtest_breakout.py's day-by-day
loop exactly, just triggered from within the finer master clock). MTF's
hourly logic is unchanged from backtest_mtf.py.

Caveat: MTF's 1H data is capped at ~2 years by Yahoo, so this combined test
only covers that window -- Breakout Hunter's OWN validation used the full
6.5-year daily history; this run says nothing about the combined portfolio
outside the ~2-year window MTF's data allows.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import strategy_breakout as strat_bo
import strategy_mtf as strat_mtf
from backtest_breakout import WARMUP_BARS as BO_WARMUP_BARS
from backtest_mtf import WARMUP_BARS as MTF_WARMUP_BARS
from data import fetch_yfinance, fetch_yfinance_1h, fetch_yfinance_4h

STARTING_EQUITY = 100_000.0
RISK_PCT = 0.01
DAILY_START = "2019-01-01"
DAILY_END = "2026-08-01"


@dataclass
class Position:
    strategy: str
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    atr_at_entry: float
    risk_amount: float
    initial_stop: float = None       # breakout only
    emergency_stop: float = None     # breakout only
    trail_active: bool = False
    trail_stop: float = None
    extreme: float = None
    exit_date: pd.Timestamp = None
    exit_price: float = None
    exit_reason: str = None

    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    def r_multiple(self) -> float:
        return self.pnl() / self.risk_amount if self.risk_amount else 0.0


def load_data(symbols):
    daily, mtf = {}, {}
    for symbol in symbols:
        try:
            daily_df = fetch_yfinance(symbol, DAILY_START, DAILY_END)
            if len(daily_df) > BO_WARMUP_BARS + 5:
                daily[symbol] = strat_bo.prepare(daily_df)
        except Exception as e:
            print(f"  skip breakout/{symbol}: {e}")
        try:
            ltf = fetch_yfinance_1h(symbol)
            htf = fetch_yfinance_4h(symbol)
            if len(ltf) > MTF_WARMUP_BARS + 5:
                mtf[symbol] = strat_mtf.prepare(ltf, htf, long_only=True)
        except Exception as e:
            print(f"  skip mtf/{symbol}: {e}")
    return daily, mtf


def run_combined(symbols, starting_equity=STARTING_EQUITY, risk_pct=RISK_PCT, verbose=True,
                  max_concurrent_positions=None, strategies=frozenset({"breakout", "mtf"})):
    daily, mtf = load_data(symbols)
    common = sorted(set(daily) & set(mtf))
    if verbose:
        print(f"{len(common)} symbols with both daily and 1H+4H data\n")

    bo_valid_from = {s: daily[s].index[BO_WARMUP_BARS] for s in common}
    mtf_valid_from = {s: mtf[s].index[MTF_WARMUP_BARS] for s in common}

    all_ts = sorted(set().union(*[set(mtf[s].index[MTF_WARMUP_BARS:]) for s in common]))

    cash = starting_equity
    positions: dict[tuple, Position] = {}   # (symbol, strategy) -> Position
    pending: dict[tuple, tuple] = {}         # (symbol, strategy) -> entry/exit payload
    closed_trades: list[Position] = []
    equity_curve = []
    last_close = {}          # symbol -> latest known price (for mark-to-market)
    last_day_seen = {}       # symbol -> date of last processed daily bar

    def equity_now():
        mtm = 0.0
        for pos in positions.values():
            px = last_close.get(pos.symbol, pos.entry_price)
            mtm += pos.shares * px
        return cash + mtm

    for ts in all_ts:
        today = ts.date()

        # ---- Breakout Hunter: fires once per day, at the first 1H tick ----
        for symbol in (common if "breakout" in strategies else []):
            if ts not in mtf[symbol].index:
                continue
            if last_day_seen.get(symbol) == today:
                continue
            last_day_seen[symbol] = today
            key = (symbol, "breakout")
            d = daily[symbol]
            prior_days = d.index[d.index.date < today]
            if len(prior_days) == 0:
                continue
            last_daily_date = prior_days[-1]
            if last_daily_date < bo_valid_from[symbol]:
                continue
            row = d.loc[last_daily_date]
            first_1h_open = float(mtf[symbol].loc[ts, "Open"])  # today's session open ~= this first 1h bar's open

            # 1. execute pending exit at today's open
            if key in pending and pending[key][0] == "exit" and key in positions:
                _, reason = pending[key]
                pos = positions.pop(key)
                pos.exit_date, pos.exit_price, pos.exit_reason = ts, first_1h_open, reason
                cash += pos.shares * first_1h_open
                closed_trades.append(pos)
                del pending[key]

            # 2. execute pending entry at today's open
            if key in pending and pending[key][0] == "entry":
                _, support, atr_sig = pending[key]
                entry_price = first_1h_open
                initial_stop = support - strat_bo.INITIAL_STOP_ATR_BUFFER * atr_sig
                emergency_stop = initial_stop - strat_bo.EMERGENCY_STOP_BUFFER_ATR * atr_sig
                stop_distance = entry_price - initial_stop
                if stop_distance > 0 and (max_concurrent_positions is None or len(positions) < max_concurrent_positions):
                    equity = equity_now()
                    risk_amount = equity * risk_pct
                    # cap by available CASH, not total equity -- with multiple
                    # simultaneous positions (2 strategies x 40 symbols), capital
                    # already committed elsewhere must not be double-spent
                    shares = min(risk_amount / stop_distance, cash / entry_price)
                    if shares > 0:
                        positions[key] = Position(
                            strategy="breakout", symbol=symbol, entry_date=ts, entry_price=entry_price,
                            shares=shares, atr_at_entry=atr_sig, risk_amount=risk_amount,
                            initial_stop=initial_stop, emergency_stop=emergency_stop,
                            trail_active=False, trail_stop=None, extreme=entry_price,
                        )
                        cash -= shares * entry_price
                del pending[key]

            # 3. manage open breakout position: daily stop check + trail update
            if key in positions:
                pos = positions[key]
                pos.extreme = max(pos.extreme, float(row["High"]))
                favorable = float(row["Close"]) - pos.entry_price
                if not pos.trail_active and favorable >= strat_bo.TRAIL_ACTIVATE_ATR * pos.atr_at_entry:
                    pos.trail_active = True
                if pos.trail_active:
                    candidate = pos.extreme - strat_bo.TRAIL_ATR_MULT * float(row["atr"])
                    pos.trail_stop = candidate if pos.trail_stop is None else max(pos.trail_stop, candidate)
                active_stop = pos.trail_stop if pos.trail_active else pos.initial_stop
                stop_reason = "trailing_stop" if pos.trail_active else "initial_stop"

                fill_price, fill_reason = None, None
                bar_open = float(row["Open"])
                if float(row["Low"]) <= pos.emergency_stop:
                    fill_price = bar_open if bar_open <= pos.emergency_stop else pos.emergency_stop
                    fill_reason = "emergency_stop"
                elif float(row["Low"]) <= active_stop:
                    fill_price = bar_open if bar_open <= active_stop else active_stop
                    fill_reason = stop_reason
                if fill_price is not None:
                    positions.pop(key)
                    pos.exit_date, pos.exit_price, pos.exit_reason = ts, fill_price, fill_reason
                    cash += pos.shares * fill_price
                    closed_trades.append(pos)

            # 4. new breakout signal (from the just-completed daily bar) -> queue for tomorrow's open
            if key not in positions and key not in pending:
                if bool(row["long_entry"]):
                    pending[key] = ("entry", float(row["support"]), float(row["atr"]))

            # 5. breakout exit condition: none beyond stops in this spec -- nothing else to queue

            last_close[symbol] = float(row["Close"])

        # ---- Multi-Timeframe RSI+Stoch: fires every 1H tick ----
        for symbol in (common if "mtf" in strategies else []):
            if ts not in mtf[symbol].index:
                continue
            if ts < mtf_valid_from[symbol]:
                continue
            row = mtf[symbol].loc[ts]
            key = (symbol, "mtf")

            if key in pending and pending[key][0] == "exit" and key in positions:
                _, reason = pending[key]
                pos = positions.pop(key)
                pos.exit_date, pos.exit_price, pos.exit_reason = ts, float(row["Open"]), reason
                cash += pos.shares * pos.exit_price
                closed_trades.append(pos)
                del pending[key]

            if key in pending and pending[key][0] == "entry":
                _, atr_sig = pending[key]
                entry_price = float(row["Open"])
                stop_distance = strat_mtf.TRAIL_ATR_MULT * atr_sig
                if stop_distance > 0 and (max_concurrent_positions is None or len(positions) < max_concurrent_positions):
                    equity = equity_now()
                    risk_amount = equity * risk_pct
                    # cap by available CASH, not total equity -- with multiple
                    # simultaneous positions (2 strategies x 40 symbols), capital
                    # already committed elsewhere must not be double-spent
                    shares = min(risk_amount / stop_distance, cash / entry_price)
                    if shares > 0:
                        positions[key] = Position(
                            strategy="mtf", symbol=symbol, entry_date=ts, entry_price=entry_price,
                            shares=shares, atr_at_entry=atr_sig, risk_amount=risk_amount,
                            trail_active=False, trail_stop=entry_price - stop_distance, extreme=entry_price,
                        )
                        cash -= shares * entry_price
                del pending[key]

            if key in positions:
                pos = positions[key]
                pos.extreme = max(pos.extreme, float(row["Close"]))
                favorable = float(row["Close"]) - pos.entry_price
                if not pos.trail_active and favorable >= strat_mtf.TRAIL_ATR_MULT * pos.atr_at_entry:
                    pos.trail_active = True
                if pos.trail_active:
                    candidate = pos.extreme - strat_mtf.TRAIL_ATR_MULT * float(row["atr"])
                    pos.trail_stop = candidate if pos.trail_stop is None else max(pos.trail_stop, candidate)

                fill_price = None
                if float(row["Low"]) <= pos.trail_stop:
                    bar_open = float(row["Open"])
                    fill_price = bar_open if bar_open <= pos.trail_stop else pos.trail_stop
                if fill_price is not None:
                    positions.pop(key)
                    pos.exit_date, pos.exit_price, pos.exit_reason = ts, fill_price, "trailing_stop"
                    cash += pos.shares * fill_price
                    closed_trades.append(pos)

            if key in positions:
                if bool(row["long_target_exit"]):
                    pending[key] = ("exit", "target_rsi_neutral")

            if key not in positions and key not in pending:
                if bool(row["long_entry"]):
                    pending[key] = ("entry", float(row["atr"]))

            last_close[symbol] = float(row["Close"])

        equity_curve.append((ts, equity_now()))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return closed_trades, equity_df
