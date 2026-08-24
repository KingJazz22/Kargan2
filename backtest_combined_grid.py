"""Shared-equity combined portfolio: Grid-Martingale (long-only -- see
FINDINGS_GRID_MARTINGALE.md, short showed no support in either the in-sample
or out-of-sample check) running alongside Breakout Hunter (long-only,
t=3.40), the only other strategy in this project with a demonstrated,
regime-robust edge as of the most recent finding in FINDINGS_MTF.md (MTF v2
flipped to a net loser on deep multi-regime history and isn't included here).

Same shared-ledger principle as backtest_combined.py: one cash pool, every
new entry (either strand) sizes off CURRENT total equity (cash + all open
positions marked to market) x risk_pct, capped by available CASH not equity
(the leverage bug backtest_combined.py found and fixed). Simpler master clock
than backtest_combined.py since both strands here are daily-only -- no mixed
1H/daily clock needed.

Grid leg-adds are the one exception to "size off current equity": a leg-add's
risk_amount is prior_leg.risk_amount x multiplier (the martingale sequence is
defined relative to the grid's OWN prior leg, not re-derived from whatever
total portfolio equity happens to be right now -- otherwise unrelated
gains/losses elsewhere in the shared portfolio would distort the grid's own
escalation sequence). The resulting share count is still cash-capped against
the shared pool's currently available cash, same as every other entry here.
"""
from dataclasses import dataclass

import pandas as pd

import strategy_grid as strat_grid
from backtest_combined import Position
from backtest_grid_martingale import (
    CREDIBLE_PCT, GRID_STOP_ATR_MULT, HALF_MARTINGALE_MULT, MAX_LEGS,
    MIN_BARS_BETWEEN_LEGS, TP_ATR_MULT, WARMUP_BARS as GRID_WARMUP_BARS,
    GridLeg, GridPosition, GridTrade, _bayes_size_multiplier, _project_breakeven,
)
from bayesian_tracker import BetaBinomialPosterior
from data import fetch_yfinance

STARTING_EQUITY = 100_000.0
RISK_PCT = 0.01
DAILY_START = "2019-01-01"
DAILY_END = "2026-08-01"


def load_data(symbols):
    prepared = {}
    for symbol in symbols:
        try:
            df = fetch_yfinance(symbol, DAILY_START, DAILY_END)
            if len(df) > GRID_WARMUP_BARS + 5:
                prepared[symbol] = strat_grid.prepare(df)  # has long_entry/short_entry/resistance/support/atr/adx/regime_kill
        except Exception as e:
            print(f"  skip {symbol}: {e}")
    return prepared


def run_combined(symbols, starting_equity=STARTING_EQUITY, risk_pct=RISK_PCT, verbose=True,
                  max_concurrent_positions=None, strategies=frozenset({"breakout", "grid"}),
                  grid_capital_cap_frac=0.2):
    """grid_capital_cap_frac: Grid's own open notional (sum of
    total_shares * weighted_avg_entry across all open grid positions) is
    capped at this fraction of current total equity -- a reserved-capital
    floor for Breakout Hunter within the SAME shared cash pool. Default 0.2
    per the sweep in FINDINGS_GRID_MARTINGALE.md (Result 5b): raising/removing
    max_concurrent_positions barely moved Breakout Hunter's combined t-stat
    (cap=20/40/None all identical, t~2.05) because cash availability, not
    position count, was the real constraint -- Grid's wider TP=2.0 target
    holds cash longer than Breakout Hunter's positions do. Reserving ~80% of
    the pool for Breakout Hunter recovered its t-stat (~1.9 -> ~2.4) AND
    improved Grid's own (~3.1 -> ~3.8) AND cost nothing on portfolio
    return/drawdown -- a genuine win-win, confirmed as the applied default
    rather than left as an opt-in. Pass None to restore the original
    unconstrained-sharing behavior."""
    data = load_data(symbols)
    if verbose:
        print(f"{len(data)} symbols loaded\n")

    valid_from = {s: data[s].index[GRID_WARMUP_BARS] for s in data}
    all_dates = sorted(set().union(*[set(data[s].index[GRID_WARMUP_BARS:]) for s in data]))

    cash = starting_equity
    positions: dict[tuple, object] = {}   # (symbol, strategy) -> Position | GridPosition
    pending: dict[tuple, tuple] = {}
    closed_trades: list = []
    equity_curve = []
    last_close = {}
    posterior = BetaBinomialPosterior()

    def equity_now():
        mtm = 0.0
        for (symbol, strategy), pos in positions.items():
            px = last_close.get(symbol, pos.legs[0].entry_price if strategy == "grid" else pos.entry_price)
            if strategy == "grid":
                mtm += pos.total_shares * px if pos.side == "long" else -pos.total_shares * px
            else:
                mtm += pos.shares * px
        return cash + mtm

    def grid_capital_budget(equity):
        """Remaining dollars Grid may still deploy right now, or None if uncapped."""
        if grid_capital_cap_frac is None:
            return None
        committed = sum(
            pos.total_shares * pos.weighted_avg_entry
            for (s, strat), pos in positions.items() if strat == "grid"
        )
        return max(grid_capital_cap_frac * equity - committed, 0.0)

    for date in all_dates:
        for symbol in data:
            if symbol not in data or date not in data[symbol].index:
                continue
            if date < valid_from[symbol]:
                continue
            row = data[symbol].loc[date]
            last_close[symbol] = float(row["Close"])

            # ---- Breakout Hunter strand ----
            if "breakout" in strategies:
                key = (symbol, "breakout")

                if key in pending and pending[key][0] == "exit" and key in positions:
                    _, reason = pending[key]
                    pos = positions.pop(key)
                    pos.exit_date, pos.exit_price, pos.exit_reason = date, float(row["Open"]), reason
                    cash += pos.shares * pos.exit_price
                    closed_trades.append(pos)
                    del pending[key]

                if key in pending and pending[key][0] == "entry":
                    _, support, atr_sig = pending[key]
                    entry_price = float(row["Open"])
                    initial_stop = support - 0.5 * atr_sig  # strategy_breakout.INITIAL_STOP_ATR_BUFFER
                    emergency_stop = initial_stop - 2.0 * atr_sig  # strategy_breakout.EMERGENCY_STOP_BUFFER_ATR
                    stop_distance = entry_price - initial_stop
                    if stop_distance > 0 and (max_concurrent_positions is None or len(positions) < max_concurrent_positions):
                        risk_amount = equity_now() * risk_pct
                        shares = min(risk_amount / stop_distance, cash / entry_price)
                        if shares > 0:
                            positions[key] = Position(
                                strategy="breakout", symbol=symbol, entry_date=date, entry_price=entry_price,
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
                    if not pos.trail_active and favorable >= 1.0 * pos.atr_at_entry:  # TRAIL_ACTIVATE_ATR
                        pos.trail_active = True
                    if pos.trail_active:
                        candidate = pos.extreme - 2.5 * float(row["atr"])  # TRAIL_ATR_MULT
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
                        pos.exit_date, pos.exit_price, pos.exit_reason = date, fill_price, fill_reason
                        cash += pos.shares * fill_price
                        closed_trades.append(pos)

                if key not in positions and key not in pending:
                    if bool(row["long_entry"]):
                        pending[key] = ("entry", float(row["support"]), float(row["atr"]))

            # ---- Grid-Martingale strand (long only, per the strategy's validated scope) ----
            if "grid" in strategies:
                key = (symbol, "grid")

                if key in pending and pending[key][0] == "exit" and key in positions:
                    _, reason = pending[key]
                    grid = positions.pop(key)
                    exit_price = float(row["Open"])
                    gt = GridTrade.from_grid(grid, date, exit_price, reason)
                    cash += grid.total_shares * exit_price
                    posterior.update(win=(reason == "take_profit"))
                    closed_trades.append(gt)
                    del pending[key]

                if key in pending and pending[key][0] == "entry":
                    _, atr_sig = pending[key]
                    entry_price = float(row["Open"])
                    stop_distance = GRID_STOP_ATR_MULT * atr_sig
                    if stop_distance > 0 and (max_concurrent_positions is None or len(positions) < max_concurrent_positions):
                        equity = equity_now()
                        risk_amount = equity * risk_pct
                        shares = min(risk_amount / stop_distance, cash / entry_price)
                        budget = grid_capital_budget(equity)
                        if budget is not None:
                            shares = min(shares, budget / entry_price)
                        if shares > 0:
                            leg0 = GridLeg(0, date, entry_price, shares, atr_sig, risk_amount)
                            grid = GridPosition(side="long")
                            grid.add_leg(leg0)
                            positions[key] = grid
                            cash -= shares * entry_price
                    del pending[key]

                if key in positions:
                    grid = positions[key]
                    grid.bars_since_last_leg += 1
                    fired = None

                    if grid.stop_hit(row):
                        bar_open = row["Open"]
                        stop_fill = bar_open if bar_open <= grid.stop_price else grid.stop_price
                        fired = ("grid_stop", stop_fill)
                    elif grid.leg_count - 1 < MAX_LEGS and grid.bars_since_last_leg >= MIN_BARS_BETWEEN_LEGS:
                        trigger_price = grid.leg_add_trigger_price()
                        if grid.leg_trigger_hit(row, trigger_price):
                            atr_now = float(row["atr"])
                            p_breakeven = _project_breakeven(grid, trigger_price, atr_now)
                            mean, ci_lo = posterior.mean, posterior.credible_lower(CREDIBLE_PCT)
                            mult = _bayes_size_multiplier(mean, ci_lo, p_breakeven)
                            if mult is not None:
                                prior_leg = grid.legs[-1]
                                risk_amount = prior_leg.risk_amount * mult  # relative to prior leg, NOT current equity
                                leg_stop_distance = GRID_STOP_ATR_MULT * atr_now
                                shares = risk_amount / leg_stop_distance
                                shares = min(shares, cash / trigger_price)  # cash-capped against the SHARED pool
                                budget = grid_capital_budget(equity_now())
                                if budget is not None:
                                    shares = min(shares, budget / trigger_price)
                                if shares > 0:
                                    leg = GridLeg(grid.leg_count, date, trigger_price, shares, atr_now, risk_amount,
                                                  bayes_mean_at_add=mean, bayes_ci_lower_at_add=ci_lo, size_mult_applied=mult)
                                    grid.add_leg(leg)
                                    cash -= shares * trigger_price
                                    fired = ("leg_add", trigger_price)

                    if fired is None and grid.tp_hit(row):
                        bar_open = row["Open"]
                        tp_fill = bar_open if bar_open >= grid.tp_price else grid.tp_price
                        fired = ("take_profit", tp_fill)

                    if fired is not None and fired[0] in ("grid_stop", "take_profit"):
                        gt = GridTrade.from_grid(grid, date, fired[1], fired[0])
                        cash += grid.total_shares * fired[1]
                        posterior.update(win=(fired[0] == "take_profit"))
                        closed_trades.append(gt)
                        positions.pop(key)

                if key in positions and bool(row["regime_kill"]):
                    pending[key] = ("exit", "regime_kill")

                if key not in positions and key not in pending:
                    if bool(row["long_entry"]):
                        pending[key] = ("entry", float(row["atr"]))

        equity_curve.append((date, equity_now()))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return closed_trades, equity_df
