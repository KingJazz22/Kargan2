"""Peak-equity circuit breaker with a timed re-arm, shared by live_trading.py
and live_trading_v2.py.

The original breaker latched: once equity fell CIRCUIT_BREAKER_DD_PCT below
its peak, new entries stopped, so only already-open positions could ever
lift equity back and un-halt it. On the 580-name fresh S&P universe that
halted the swing strategy permanently in 2020 (CAGR 1.6% vs 12.6% without
the breaker -- FINDINGS_PCSE_FRESH_UNIVERSE.md). Now: after COOLDOWN_DAYS
halted, the breaker re-arms (peak resets to current equity) and entries
resume at REARM_RISK_MULT x risk until equity regains the pre-trip peak.
Ledger check of the variants: sweep_risk_levels.CIRCUIT_BREAKER_COOLDOWN_DAYS.

State keys (persisted in the runner's JSON state):
  breaker_tripped_at       ISO date the current halt started
  breaker_recovery_target  pre-trip peak still to be regained after a re-arm
"""
from datetime import date

COOLDOWN_DAYS = 30
REARM_RISK_MULT = 0.5


def check(state, equity, dd_pct, today=None):
    """Updates state["peak_equity"] and the breaker keys. Returns
    (halt_entries, risk_mult): risk_mult < 1 while recovering after a re-arm."""
    today = today or date.today()
    peak = max(state.get("peak_equity", equity), equity)
    state["peak_equity"] = peak
    drawdown = (peak - equity) / peak if peak > 0 else 0.0

    halt = False
    if drawdown >= dd_pct:
        tripped = date.fromisoformat(state.setdefault("breaker_tripped_at", today.isoformat()))
        halted_days = (today - tripped).days
        if halted_days >= COOLDOWN_DAYS:
            state["breaker_recovery_target"] = max(state.get("breaker_recovery_target") or 0.0, peak)
            state["peak_equity"] = equity
            state.pop("breaker_tripped_at")
            print(f"\n!!! CIRCUIT BREAKER RE-ARMED after {halted_days} days halted: peak reset to ${equity:,.2f}; "
                  f"entries resume at {REARM_RISK_MULT:.0%} risk until equity regains ${state['breaker_recovery_target']:,.2f}")
        else:
            print(f"\n!!! CIRCUIT BREAKER: equity ${equity:,.2f} is {drawdown:.1%} below peak ${peak:,.2f} -- "
                  f"new entries halted (day {halted_days} of {COOLDOWN_DAYS} before re-arm)")
            halt = True
    else:
        state.pop("breaker_tripped_at", None)

    target = state.get("breaker_recovery_target")
    if target is not None and equity >= target:
        print(f"  circuit-breaker recovery complete: equity ${equity:,.2f} regained ${target:,.2f} -- full risk restored")
        state.pop("breaker_recovery_target")
        target = None
    return halt, (REARM_RISK_MULT if target is not None else 1.0)
