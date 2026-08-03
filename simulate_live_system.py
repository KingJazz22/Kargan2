"""Simulates live_trading.py's EXACT configuration on historical data:
Breakout Hunter (daily) + MTF v2 (4H execution + Daily confirm, TSL-only exit),
same 40 symbols (US_SYMBOLS + OOS_US_SYMBOLS), same $9,995.18 starting equity
(the real current "Kargan 2" account balance), same 1% risk/trade, same
MAX_CONCURRENT_POSITIONS=10 cap, sharing one capital pool.

Master clock is the union of 4H timestamps (MTF's native granularity, same as
how live_trading.py itself is scheduled to run every ~2h). Breakout Hunter's
daily logic fires once per day at the first 4H tick of each new trading day,
using the just-completed prior daily bar -- same pattern as backtest_combined.py,
just re-gridded onto 4H ticks instead of 1H since live_trading.py's MTF uses
4H+Daily now (v2), not the original 1H+4H (v1).

Data is yfinance hourly resampled to 4H, capped at ~2 years of history by
Yahoo -- same limitation as every other MTF backtest in this project.
"""
from dataclasses import dataclass

import pandas as pd

import strategy_breakout as strat_bo
import strategy_mtf as strat_mtf
from backtest_breakout import WARMUP_BARS as BO_WARMUP_BARS
from backtest_mtf import WARMUP_BARS as MTF_WARMUP_BARS
from basket import US_SYMBOLS
from test_mtf_oos_symbols import OOS_US_SYMBOLS
from alpaca_fetch import fetch_stock_daily, fetch_stock_4h

SYMBOLS = US_SYMBOLS + OOS_US_SYMBOLS
STARTING_EQUITY = 9995.18
RISK_PCT = 0.01
MAX_CONCURRENT_POSITIONS = 10
DAILY_START = "2016-01-01"  # Alpaca's equity hourly/4H history starts here -- using
DAILY_END = "2026-08-01"    # the same start for daily keeps both series' windows aligned

MTF_PARAMS = {"long_only": True, "htf_bar_hours": 24, "tsl_only_exit": True}


@dataclass
class Position:
    strategy: str
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    atr_at_entry: float
    risk_amount: float
    initial_stop: float = None
    emergency_stop: float = None
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


def fetch_raw(symbols, raw_cache=None):
    """Fetches (daily_df, 4h_df) once per symbol. Pass the same raw_cache dict
    across multiple load_data()/run_combined() calls (e.g. a filter sweep) to
    avoid re-hitting Alpaca for data that doesn't depend on the MTF filter."""
    if raw_cache is None:
        raw_cache = {}
    for symbol in symbols:
        if symbol in raw_cache:
            continue
        daily_raw, ltf_raw = None, None
        try:
            daily_raw = fetch_stock_daily(symbol, DAILY_START, DAILY_END)
        except Exception as e:
            print(f"  skip fetch daily/{symbol}: {e}")
        try:
            ltf_raw = fetch_stock_4h(symbol, DAILY_START, DAILY_END)
        except Exception as e:
            print(f"  skip fetch 4h/{symbol}: {e}")
        raw_cache[symbol] = (daily_raw, ltf_raw)
    return raw_cache


def load_data(symbols, mtf_trend_filter=None, raw_cache=None):
    """mtf_trend_filter: None (off) or (filter_type, period), e.g. ("sma", 200)."""
    raw_cache = fetch_raw(symbols, raw_cache)
    daily, mtf = {}, {}
    for symbol in symbols:
        daily_raw, ltf_raw = raw_cache.get(symbol, (None, None))
        if daily_raw is not None and len(daily_raw) > BO_WARMUP_BARS + 5:
            daily[symbol] = strat_bo.prepare(daily_raw)
        if ltf_raw is not None and daily_raw is not None and len(ltf_raw) > MTF_WARMUP_BARS + 5:
            params = dict(MTF_PARAMS)
            if mtf_trend_filter is not None:
                filter_type, period = mtf_trend_filter
                # Only take long entries above the trend line -- turns MTF from
                # "buy any oversold reading" into "buy dips within an uptrend."
                # daily_raw doubles as the trend-filter source (same 24h bar
                # shift strategy_mtf.py already applies for lookahead safety).
                params["trend_filter_df"] = daily_raw
                params["trend_filter_period"] = period
                params["trend_filter_bar_hours"] = 24
                params["trend_filter_type"] = filter_type
            mtf[symbol] = strat_mtf.prepare(ltf_raw, daily_raw, **params)
    return daily, mtf


def run_combined(symbols, starting_equity, risk_pct, max_concurrent_positions, verbose=True,
                  mtf_trend_filter=None, raw_cache=None):
    daily, mtf = load_data(symbols, mtf_trend_filter=mtf_trend_filter, raw_cache=raw_cache)
    common = sorted(set(daily) & set(mtf))
    if verbose:
        print(f"{len(common)}/{len(symbols)} symbols with both daily and 4H+Daily data\n")

    bo_valid_from = {s: daily[s].index[BO_WARMUP_BARS] for s in common}
    mtf_valid_from = {s: mtf[s].index[MTF_WARMUP_BARS] for s in common}

    all_ts = sorted(set().union(*[set(mtf[s].index[MTF_WARMUP_BARS:]) for s in common]))

    cash = starting_equity
    positions: dict[tuple, Position] = {}
    pending: dict[tuple, tuple] = {}
    closed_trades: list[Position] = []
    equity_curve = []
    last_close = {}
    last_day_seen = {}

    def equity_now():
        mtm = 0.0
        for pos in positions.values():
            px = last_close.get(pos.symbol, pos.entry_price)
            mtm += pos.shares * px
        return cash + mtm

    for ts in all_ts:
        today = ts.date()

        # ---- Breakout Hunter: fires once per day, at the first 4H tick ----
        for symbol in common:
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
            first_4h_open = float(mtf[symbol].loc[ts, "Open"])

            if key in pending and pending[key][0] == "exit" and key in positions:
                _, reason = pending[key]
                pos = positions.pop(key)
                pos.exit_date, pos.exit_price, pos.exit_reason = ts, first_4h_open, reason
                cash += pos.shares * first_4h_open
                closed_trades.append(pos)
                del pending[key]

            if key in pending and pending[key][0] == "entry":
                _, support, atr_sig = pending[key]
                entry_price = first_4h_open
                initial_stop = support - strat_bo.INITIAL_STOP_ATR_BUFFER * atr_sig
                emergency_stop = initial_stop - strat_bo.EMERGENCY_STOP_BUFFER_ATR * atr_sig
                stop_distance = entry_price - initial_stop
                if stop_distance > 0 and (max_concurrent_positions is None or len(positions) < max_concurrent_positions):
                    equity = equity_now()
                    risk_amount = equity * risk_pct
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
                if float(row["Low"]) <= pos.emergency_stop:
                    fill_price, fill_reason = pos.emergency_stop, "emergency_stop"
                elif float(row["Low"]) <= active_stop:
                    fill_price, fill_reason = active_stop, stop_reason
                if fill_price is not None:
                    positions.pop(key)
                    pos.exit_date, pos.exit_price, pos.exit_reason = ts, fill_price, fill_reason
                    cash += pos.shares * fill_price
                    closed_trades.append(pos)

            if key not in positions and key not in pending:
                if bool(row["long_entry"]):
                    pending[key] = ("entry", float(row["support"]), float(row["atr"]))

            last_close[symbol] = float(row["Close"])

        # ---- MTF v2: fires every 4H tick, TSL-only exit ----
        for symbol in common:
            if ts not in mtf[symbol].index:
                continue
            if ts < mtf_valid_from[symbol]:
                continue
            row = mtf[symbol].loc[ts]
            key = (symbol, "mtf")

            if key in pending and pending[key][0] == "entry":
                _, atr_sig = pending[key]
                entry_price = float(row["Open"])
                stop_distance = strat_mtf.TRAIL_ATR_MULT * atr_sig
                if stop_distance > 0 and (max_concurrent_positions is None or len(positions) < max_concurrent_positions):
                    equity = equity_now()
                    risk_amount = equity * risk_pct
                    shares = min(risk_amount / stop_distance, cash / entry_price)
                    if shares > 0:
                        positions[key] = Position(
                            strategy="mtf", symbol=symbol, entry_date=ts, entry_price=entry_price,
                            shares=shares, atr_at_entry=atr_sig, risk_amount=risk_amount,
                            trail_active=True, trail_stop=entry_price - stop_distance, extreme=entry_price,
                        )
                        cash -= shares * entry_price
                del pending[key]

            if key in positions:
                pos = positions[key]
                pos.extreme = max(pos.extreme, float(row["Close"]))
                candidate = pos.extreme - strat_mtf.TRAIL_ATR_MULT * float(row["atr"])
                pos.trail_stop = candidate if pos.trail_stop is None else max(pos.trail_stop, candidate)

                fill_price = None
                if float(row["Low"]) <= pos.trail_stop:
                    fill_price = pos.trail_stop
                if fill_price is not None:
                    positions.pop(key)
                    pos.exit_date, pos.exit_price, pos.exit_reason = ts, fill_price, "trailing_stop"
                    cash += pos.shares * fill_price
                    closed_trades.append(pos)

            if key not in positions and key not in pending:
                if bool(row["long_entry"]):
                    pending[key] = ("entry", float(row["atr"]))

            last_close[symbol] = float(row["Close"])

        equity_curve.append((ts, equity_now()))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return closed_trades, equity_df


def summarize(closed_trades, equity_df, starting_equity):
    import numpy as np

    print(f"\n{'='*60}")
    print("LIVE SYSTEM SIMULATION -- Breakout Hunter + MTF v2 (4H+Daily)")
    print(f"{'='*60}")
    print(f"Symbols: {len(SYMBOLS)} | Starting equity: ${starting_equity:,.2f} | "
          f"Risk/trade: {RISK_PCT:.1%} | Max concurrent positions: {MAX_CONCURRENT_POSITIONS}")

    if equity_df.empty:
        print("No data / no bars processed.")
        return

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / starting_equity - 1) * 100
    days = (equity_df.index[-1] - equity_df.index[0]).days
    years = days / 365.25
    cagr = ((final_equity / starting_equity) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - running_max) / running_max
    max_dd = drawdown.min() * 100

    daily_eq = equity_df["equity"].resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0.0

    print(f"\nWindow: {equity_df.index[0].date()} -> {equity_df.index[-1].date()} ({years:.1f} yrs)")
    print(f"Final equity: ${final_equity:,.2f}")
    print(f"Total return: {total_return:+.1f}%")
    print(f"CAGR: {cagr:+.1f}%")
    print(f"Max drawdown: {max_dd:.1f}%")
    print(f"Sharpe (daily, annualized): {sharpe:.2f}")

    if closed_trades:
        n = len(closed_trades)
        wins = [t for t in closed_trades if t.pnl() > 0]
        r_multiples = np.array([t.r_multiple() for t in closed_trades])
        total_pnl = sum(t.pnl() for t in closed_trades)
        print(f"\nClosed trades: {n} | Win rate: {len(wins)/n:.1%}")
        print(f"Avg R: {r_multiples.mean():+.3f} | Total realized PnL: ${total_pnl:+,.2f}")

        for strat_name in ("breakout", "mtf"):
            trades = [t for t in closed_trades if t.strategy == strat_name]
            if not trades:
                continue
            wr = sum(1 for t in trades if t.pnl() > 0) / len(trades)
            avg_r = np.mean([t.r_multiple() for t in trades])
            pnl = sum(t.pnl() for t in trades)
            print(f"  {strat_name:>9}: {len(trades)} trades | win rate {wr:.1%} | avg R {avg_r:+.3f} | PnL ${pnl:+,.2f}")
    else:
        print("\nNo trades closed.")


if __name__ == "__main__":
    import sys
    trend_filter = ("sma", 200) if "--mtf-trend-filter" in sys.argv else None
    print(f"MTF trend filter: {trend_filter if trend_filter else 'OFF (baseline)'}")
    trades, eq_df = run_combined(SYMBOLS, STARTING_EQUITY, RISK_PCT, MAX_CONCURRENT_POSITIONS,
                                  mtf_trend_filter=trend_filter)
    summarize(trades, eq_df, STARTING_EQUITY)
