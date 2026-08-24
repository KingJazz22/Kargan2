"""Event-driven backtester for the Bayesian-Gated Half-Martingale Grid
strategy. Long and short.

Leg 0 enters on Breakout Hunter's squeeze+breakout(/breakdown) signal
(strategy_grid.py, which wraps strategy_breakout.py unchanged). If price then
moves AGAINST the position by GRID_STEP_ATR x ATR (measured off the last
leg's own locked-in ATR snapshot -- a resting trigger level, not recomputed
every bar) and at least MIN_BARS_BETWEEN_LEGS bars have passed since the last
leg, a new leg may be added: sized at the prior leg's risk x
HALF_MARTINGALE_MULT, further scaled down by how confidently a shared,
whole-basket Bayesian posterior believes this grid will resolve via
take-profit rather than stop/kill (see _bayes_size_multiplier). Exit is
either the take-profit (weighted-avg entry +/- a small ATR multiple), the
hard grid-stop (beyond the worst leg's entry by GRID_STOP_ATR_MULT x ATR --
bounds the loss, no further averaging past it), or a regime-kill (ADX drops
below strategy.ADX_WEAK, "dead trend").

Same lookahead-safe convention as every other engine here: signals read on a
bar's close execute at the next bar's open; stop/leg-add/take-profit levels
are resting orders that fill intrabar off that bar's High/Low. Within a bar,
at most one structural event fires, priority grid-stop > leg-add > take-profit
(the "more adverse outcome wins ties" rule already used for
emergency_stop-before-active_stop in backtest_breakout.py).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import strategy_breakout as strat_bo
import strategy_grid as strat
from bayesian_tracker import BetaBinomialPosterior

WARMUP_BARS = strat_bo.PERCENTILE_LOOKBACK + 50

BASE_RISK_PCT = 0.005
HALF_MARTINGALE_MULT = 1.4
GRID_STEP_ATR = 1.0
MAX_LEGS = 3  # additional legs beyond leg 0 (4 total per grid)
MIN_BARS_BETWEEN_LEGS = 1  # see sweep_grid_legtiming.py -- avg R and win rate both improve monotonically
                            # as this shrinks toward 0; note bars_since_last_leg increments BEFORE the
                            # gate check each bar, so in practice 0 and 1 behave identically (a same-day
                            # second leg is possible on a high-range entry day either way)
TP_ATR_MULT = 2.0  # see sweep_grid_tp.py -- 0.75 lost money out-of-sample (-6.9% total return);
                    # 2.0 is the first level where BOTH in-sample and out-of-sample are net positive
GRID_STOP_ATR_MULT = 3.0
CREDIBLE_PCT = 0.20

assert GRID_STOP_ATR_MULT > GRID_STEP_ATR, (
    "grid-stop must sit further away than the leg-add trigger, or a leg-add "
    "would never get a chance to fire before the grid-stop already has"
)


@dataclass
class GridLeg:
    leg_index: int
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    atr_at_entry: float
    risk_amount: float
    bayes_mean_at_add: float = None
    bayes_ci_lower_at_add: float = None
    size_mult_applied: float = 1.0
    cash_capped: bool = False


@dataclass
class GridPosition:
    side: str  # "long" or "short"
    tp_atr_mult: float = TP_ATR_MULT
    grid_step_atr: float = GRID_STEP_ATR
    grid_stop_atr_mult: float = GRID_STOP_ATR_MULT
    legs: list = field(default_factory=list)
    total_shares: float = 0.0
    weighted_avg_entry: float = 0.0
    last_leg_entry: float = None
    atr_at_last_leg: float = None
    stop_price: float = None
    tp_price: float = None
    bars_since_last_leg: int = 0

    def add_leg(self, leg: GridLeg) -> None:
        new_total = self.total_shares + leg.shares
        self.weighted_avg_entry = (
            self.weighted_avg_entry * self.total_shares + leg.entry_price * leg.shares
        ) / new_total
        self.total_shares = new_total
        self.last_leg_entry = leg.entry_price
        self.atr_at_last_leg = leg.atr_at_entry
        self.legs.append(leg)
        self.bars_since_last_leg = 0

        if self.side == "long":
            self.stop_price = self.last_leg_entry - self.grid_stop_atr_mult * leg.atr_at_entry
            self.tp_price = self.weighted_avg_entry + self.tp_atr_mult * leg.atr_at_entry
        else:
            self.stop_price = self.last_leg_entry + self.grid_stop_atr_mult * leg.atr_at_entry
            self.tp_price = self.weighted_avg_entry - self.tp_atr_mult * leg.atr_at_entry

    @property
    def leg_count(self) -> int:
        return len(self.legs)

    def leg_add_trigger_price(self) -> float:
        if self.side == "long":
            return self.last_leg_entry - self.grid_step_atr * self.atr_at_last_leg
        return self.last_leg_entry + self.grid_step_atr * self.atr_at_last_leg

    def stop_hit(self, row) -> bool:
        return row["Low"] <= self.stop_price if self.side == "long" else row["High"] >= self.stop_price

    def leg_trigger_hit(self, row, trigger_price) -> bool:
        return row["Low"] <= trigger_price if self.side == "long" else row["High"] >= trigger_price

    def tp_hit(self, row) -> bool:
        return row["High"] >= self.tp_price if self.side == "long" else row["Low"] <= self.tp_price


@dataclass
class GridTrade:
    side: str
    entry_date: pd.Timestamp
    weighted_avg_entry: float
    total_shares: float
    num_legs: int
    legs: list
    risk_amount: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str

    def pnl(self) -> float:
        diff = self.exit_price - self.weighted_avg_entry
        if self.side == "short":
            diff = -diff
        return diff * self.total_shares

    def r_multiple(self) -> float:
        return self.pnl() / self.risk_amount if self.risk_amount else 0.0

    @classmethod
    def from_grid(cls, grid: GridPosition, exit_date, exit_price, exit_reason) -> "GridTrade":
        # Final realized stop distance -> a grid_stop exit lands at exactly
        # -1R by construction, matching every other strategy's Trade
        # convention (see backtest.py/backtest_breakout.py).
        stop_distance = abs(grid.weighted_avg_entry - grid.stop_price)
        risk_amount = grid.total_shares * stop_distance
        return cls(
            side=grid.side, entry_date=grid.legs[0].entry_date,
            weighted_avg_entry=grid.weighted_avg_entry, total_shares=grid.total_shares,
            num_legs=grid.leg_count, legs=list(grid.legs), risk_amount=risk_amount,
            exit_date=exit_date, exit_price=exit_price, exit_reason=exit_reason,
        )


def _project_breakeven(grid: GridPosition, candidate_price: float, atr_now: float) -> float:
    """Pro-forma breakeven win-probability for adding a leg, assuming it fills
    at FULL (unshrunk) size -- breaks the circular dependency where the gate
    needs the payoff ratio but the payoff ratio needs the (gate-dependent)
    shares. Only this nominal projection feeds the gate; actual shares (after
    any confidence-shrink) are computed separately by the caller."""
    prior = grid.legs[-1]
    nominal_shares = (prior.risk_amount * HALF_MARTINGALE_MULT) / (grid.grid_stop_atr_mult * atr_now)
    new_total = grid.total_shares + nominal_shares
    new_wavg = (grid.weighted_avg_entry * grid.total_shares + candidate_price * nominal_shares) / new_total

    stop_distance = grid.grid_stop_atr_mult * atr_now
    if grid.side == "long":
        tp_distance = (new_wavg + grid.tp_atr_mult * atr_now) - candidate_price
    else:
        tp_distance = candidate_price - (new_wavg - grid.tp_atr_mult * atr_now)
    tp_distance = max(tp_distance, 1e-9)
    return stop_distance / (stop_distance + tp_distance)


def _bayes_size_multiplier(mean: float, ci_lower: float, p_breakeven: float):
    """Returns the escalation multiplier to apply to the prior leg's
    risk_amount, or None to hard-deny the leg add."""
    if mean <= p_breakeven:
        return None
    if ci_lower > p_breakeven:
        return HALF_MARTINGALE_MULT
    confidence_frac = np.clip((ci_lower - p_breakeven) / (mean - p_breakeven), 0.0, 1.0)
    return 1.0 + confidence_frac * (HALF_MARTINGALE_MULT - 1.0)


def run_backtest(df: pd.DataFrame, starting_equity: float = 100_000.0,
                  risk_pct: float = BASE_RISK_PCT, posterior: BetaBinomialPosterior = None,
                  long_only: bool = True, min_bars_between_legs: int = MIN_BARS_BETWEEN_LEGS,
                  tp_atr_mult: float = TP_ATR_MULT, grid_step_atr: float = GRID_STEP_ATR,
                  grid_stop_atr_mult: float = GRID_STOP_ATR_MULT):
    assert grid_stop_atr_mult > grid_step_atr, (
        "grid-stop must sit further away than the leg-add trigger, or a leg-add "
        "would never get a chance to fire before the grid-stop already has"
    )
    data = strat.prepare(df)
    n = len(data)
    if n <= WARMUP_BARS + 5:
        raise ValueError(f"Not enough bars ({n}) for warmup ({WARMUP_BARS})")

    posterior = posterior if posterior is not None else BetaBinomialPosterior()

    cash = starting_equity
    grid: GridPosition = None
    pending_entry = None       # ("long"|"short", atr_at_signal)
    pending_exit_reason = None
    trades: list[GridTrade] = []
    equity_curve = []

    rows = data.iloc[WARMUP_BARS:]

    for date, row in rows.iterrows():
        # 1. execute pending grid-close (regime_kill, queued from prior bar's close) at today's open
        if grid is not None and pending_exit_reason is not None:
            exit_price = row["Open"]
            gt = GridTrade.from_grid(grid, date, exit_price, pending_exit_reason)
            if grid.side == "long":
                cash += grid.total_shares * exit_price
            else:
                cash -= grid.total_shares * exit_price
            posterior.update(win=(pending_exit_reason == "take_profit"))
            trades.append(gt)
            grid, pending_exit_reason = None, None

        # 2. execute pending fresh leg-0 entry at today's open
        if grid is None and pending_entry is not None:
            side, atr_sig = pending_entry
            entry_price = row["Open"]
            stop_distance = grid_stop_atr_mult * atr_sig
            if stop_distance > 0:
                risk_amount = cash * risk_pct
                shares = min(risk_amount / stop_distance, cash / entry_price)
                if shares > 0:
                    leg0 = GridLeg(0, date, entry_price, shares, atr_sig, risk_amount)
                    grid = GridPosition(side=side, tp_atr_mult=tp_atr_mult,
                                         grid_step_atr=grid_step_atr, grid_stop_atr_mult=grid_stop_atr_mult)
                    grid.add_leg(leg0)
                    if side == "long":
                        cash -= shares * entry_price
                    else:
                        cash += shares * entry_price
            pending_entry = None

        # 3. manage open grid: intrabar fills, priority grid-stop > leg-add > take-profit
        if grid is not None:
            grid.bars_since_last_leg += 1
            fired = None

            if grid.stop_hit(row):
                bar_open = row["Open"]
                if grid.side == "long":
                    stop_fill = bar_open if bar_open <= grid.stop_price else grid.stop_price
                else:
                    stop_fill = bar_open if bar_open >= grid.stop_price else grid.stop_price
                fired = ("grid_stop", stop_fill)

            elif grid.leg_count - 1 < MAX_LEGS and grid.bars_since_last_leg >= min_bars_between_legs:
                trigger_price = grid.leg_add_trigger_price()
                if grid.leg_trigger_hit(row, trigger_price):
                    atr_now = float(row["atr"])
                    p_breakeven = _project_breakeven(grid, trigger_price, atr_now)
                    mean, ci_lo = posterior.mean, posterior.credible_lower(CREDIBLE_PCT)
                    mult = _bayes_size_multiplier(mean, ci_lo, p_breakeven)
                    if mult is not None:
                        prior_leg = grid.legs[-1]
                        risk_amount = prior_leg.risk_amount * mult
                        stop_distance = GRID_STOP_ATR_MULT * atr_now
                        shares = risk_amount / stop_distance
                        cash_capped = shares > cash / trigger_price
                        shares = min(shares, cash / trigger_price)
                        if shares > 0:
                            leg = GridLeg(
                                grid.leg_count, date, trigger_price, shares, atr_now, risk_amount,
                                bayes_mean_at_add=mean, bayes_ci_lower_at_add=ci_lo,
                                size_mult_applied=mult, cash_capped=cash_capped,
                            )
                            grid.add_leg(leg)
                            if grid.side == "long":
                                cash -= shares * trigger_price
                            else:
                                cash += shares * trigger_price
                            fired = ("leg_add", trigger_price)

            if fired is None and grid.tp_hit(row):
                bar_open = row["Open"]
                if grid.side == "long":
                    tp_fill = bar_open if bar_open >= grid.tp_price else grid.tp_price
                else:
                    tp_fill = bar_open if bar_open <= grid.tp_price else grid.tp_price
                fired = ("take_profit", tp_fill)

            if fired is not None and fired[0] in ("grid_stop", "take_profit"):
                gt = GridTrade.from_grid(grid, date, fired[1], fired[0])
                if grid.side == "long":
                    cash += grid.total_shares * fired[1]
                else:
                    cash -= grid.total_shares * fired[1]
                posterior.update(win=(fired[0] == "take_profit"))
                trades.append(gt)
                grid = None

        # 4. close-based regime-kill -> queue for next bar's open
        if grid is not None and bool(row["regime_kill"]):
            pending_exit_reason = "regime_kill"

        # 5. close-based fresh leg-0 entry -> queue for next bar's open (only if flat)
        if grid is None and pending_entry is None:
            if bool(row["long_entry"]):
                pending_entry = ("long", float(row["atr"]))
            elif not long_only and bool(row["short_entry"]):
                pending_entry = ("short", float(row["atr"]))

        # 6. mark-to-market equity
        if grid is not None:
            mtm = grid.total_shares * row["Close"] if grid.side == "long" else -grid.total_shares * row["Close"]
            equity = cash + mtm
        else:
            equity = cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df
