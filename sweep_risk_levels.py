"""Sweeps RISK_PCT across the three strategies actually live in
live_trading.py right now -- Breakout Hunter (daily) + Swing-Structure
Breakout (daily) + PCSE (4H) -- MTF entries are disabled and Orland/EWZ runs
on its own separate small-risk sleeve/account, so neither is part of this
capital pool (see live_trading.py's own module docstring).

Two layers, answering two different questions:

1. HISTORICAL REPLAY (one run per risk_pct): trades are generated once per
   strategy/symbol with unlimited equity and no cap (decoupled from capital,
   same pattern as simulate_full_kargan2_system.py), then replayed together
   through ONE shared ledger with REAL cash-constrained sizing and the SAME
   account-level risk management live_trading.py actually runs -- no position
   cap (removed live, see git history), a 15%-drawdown circuit breaker, a
   4%/day loss gate, and the rolling-expectancy risk throttle (0.7x sizing
   after 3 losses in a row or negative 20-trade expectancy). This gives real
   DD/return/Sharpe/trade-frequency numbers, but it is still only the ONE
   historical path that actually happened.

2. TRADE-SEQUENCE MONTE CARLO (per risk_pct): a single historical path can't
   show what bad luck in trade ORDERING could have done to the same edge --
   whether the real losses had clustered early instead of late, this same
   set of trades could have produced a much worse drawdown. Reshuffles the
   realized R-multiples thousands of times under simple fixed-fractional
   compounding (equity *= 1 + risk_pct * r), preserving the risk throttle's
   reaction to streaks. This layer deliberately ignores calendar overlap/
   concurrency (a shuffled sequence has no calendar) -- it is a standard,
   simplified risk-of-ruin technique, not a replacement for layer 1's
   concurrency-aware replay. See the caveats printed at the end of the run.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_breakout as bt_bo
import backtest_swing_breakout as bt_swing
import backtest_pcse_final as bt_pcse
from basket import US_SYMBOLS
from test_mtf_oos_symbols import OOS_US_SYMBOLS
from alpaca_fetch import fetch_stock_daily, fetch_stock_4h

SYMBOLS = US_SYMBOLS + OOS_US_SYMBOLS          # breakout + swing_breakout universe (40 symbols)
PCSE_SYMBOLS = US_SYMBOLS                       # PCSE was only ever validated on this 20-symbol half

STARTING_EQUITY = 9995.18                       # same baseline used throughout this project's simulations
CIRCUIT_BREAKER_DD_PCT = 0.15
DAILY_LOSS_LIMIT_PCT = 0.04
RISK_THROTTLE_MULT = 0.7
RISK_THROTTLE_LOOKBACK = 20
RISK_THROTTLE_CONSEC_LOSSES = 3

RISK_LEVELS = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
N_MONTE_CARLO = 3000
RUIN_THRESHOLDS = [0.30, 0.50, 0.70, 0.90]      # "chance of exploding" = P(max DD >= threshold)

CACHE_FILE = Path("_sweep_risk_levels_bars_cache.pkl")


def to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_bars_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_bars_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def build_trades():
    """Decoupled trade generation: each strategy/symbol backtested alone with
    effectively unlimited equity (1e12) and its own single-position-per-symbol
    engine, at the SAME risk_pct throughout (1%) -- the only thing that
    matters out of this step is entry/exit price/time and stop distance
    (R-sizing), which is risk_pct-independent; actual sizing happens later in
    the shared-ledger replay at whatever risk_pct is being tested."""
    cache = load_bars_cache()
    all_trades = []

    print("=== Breakout Hunter + Swing-Structure Breakout (daily) ===")
    for symbol in SYMBOLS:
        key = ("daily", symbol)
        if key not in cache:
            try:
                cache[key] = fetch_stock_daily(symbol)
            except Exception as e:
                print(f"  skip {symbol} daily: {e}")
                cache[key] = None
        df = cache[key]
        if df is None:
            continue

        if len(df) > bt_bo.WARMUP_BARS + 5:
            trades, _ = bt_bo.run_backtest(df, starting_equity=1e12, risk_pct=0.01)
            for t in trades:
                if t.exit_price is None:
                    continue
                all_trades.append({
                    "strategy_key": "breakout", "symbol": symbol,
                    "entry_time": to_utc(t.entry_date), "exit_time": to_utc(t.exit_date),
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "pnl_pct": t.exit_price / t.entry_price - 1,
                    "stop_distance": t.entry_price - t.initial_stop,
                })
        if len(df) > bt_swing.WARMUP_BARS + 5:
            trades, _ = bt_swing.run_backtest(df, starting_equity=1e12, risk_pct=0.01)
            for t in trades:
                if t.exit_price is None:
                    continue
                all_trades.append({
                    "strategy_key": "swing_breakout", "symbol": symbol,
                    "entry_time": to_utc(t.entry_date), "exit_time": to_utc(t.exit_date),
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "pnl_pct": t.exit_price / t.entry_price - 1,
                    "stop_distance": t.entry_price - t.initial_stop,
                })
    n_bo = sum(1 for t in all_trades if t["strategy_key"] == "breakout")
    n_sw = sum(1 for t in all_trades if t["strategy_key"] == "swing_breakout")
    print(f"  {n_bo} breakout trades, {n_sw} swing_breakout trades\n")

    print("=== PCSE (4H, SL=12x / TP=BB(3.0std)) ===")
    n_pcse = 0
    for symbol in PCSE_SYMBOLS:
        key = ("4h", symbol)
        if key not in cache:
            try:
                cache[key] = fetch_stock_4h(symbol)
            except Exception as e:
                print(f"  skip {symbol} 4h: {e}")
                cache[key] = None
        df = cache[key]
        if df is None:
            continue
        if len(df) <= bt_pcse.WARMUP_BARS + 5:
            continue
        try:
            trades, _ = bt_pcse.run_backtest(df, starting_equity=1e12, risk_pct=0.01)
        except Exception as e:
            print(f"  skip {symbol} pcse: {e}")
            continue
        for t in trades:
            if t.exit_price is None:
                continue
            pnl_pct = (t.exit_price / t.entry_price - 1) * (1 if t.side == "long" else -1)
            all_trades.append({
                "strategy_key": "pcse", "symbol": symbol,
                "entry_time": to_utc(t.entry_date), "exit_time": to_utc(t.exit_date),
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "pnl_pct": pnl_pct, "stop_distance": t.stop_distance,
            })
        n_pcse += sum(1 for t in trades if t.exit_price is not None)
    print(f"  {n_pcse} PCSE trades\n")

    save_bars_cache(cache)
    all_trades.sort(key=lambda t: t["entry_time"])
    print(f"Total combined trades (unlimited-equity generation): {len(all_trades)}\n")
    return all_trades


# --------------------------------------------------------------------------
# Layer 1: concurrency-aware shared-ledger replay with live_trading.py's real
# account-level risk management (circuit breaker / daily loss gate / risk
# throttle), no position cap (matches the current live config).
# --------------------------------------------------------------------------

def replay_shared_ledger(all_trades, risk_pct):
    events = []
    by_id = {}
    for i, t in enumerate(all_trades):
        tid = f"t{i}"
        by_id[tid] = t
        # exits sort before entries at the SAME timestamp (frees capital first)
        # -- except a same-bar entry+exit (stopped out the day it entered),
        # where that ordering would try to exit a position never entered.
        # Give that exit priority 2 so its own entry (priority 1) goes first.
        same_bar = t["entry_time"] == t["exit_time"]
        events.append((t["entry_time"], 1, tid))
        events.append((t["exit_time"], 2 if same_bar else 0, tid))
    events.sort(key=lambda e: (e[0], e[1]))

    cash = STARTING_EQUITY
    open_positions = {}          # tid -> dict(notional, entry_price, pnl_pct, strategy_key)
    equity_curve = []
    closed = []                  # chronological, for the risk throttle + Monte Carlo source
    peak_equity = STARTING_EQUITY
    daily_start_date = None
    daily_start_equity = STARTING_EQUITY
    risk_throttled = False
    entries_blocked_circuit = 0
    entries_blocked_daily = 0
    entries_taken = 0

    def equity_now():
        mtm = sum(p["notional"] * (1 + p["pnl_pct"]) for p in open_positions.values())
        return cash + mtm

    for ts, _, tid in events:
        t = by_id[tid]
        is_exit = tid in open_positions and t["exit_time"] == ts
        is_entry = tid not in open_positions and t["entry_time"] == ts

        if is_exit:
            pos = open_positions.pop(tid)
            pnl_dollars = pos["notional"] * pos["pnl_pct"]
            cash += pos["notional"] + pnl_dollars
            r_multiple = pnl_dollars / pos["risk_amount"] if pos["risk_amount"] else 0.0
            closed.append({"_tid": tid, "strategy_key": pos["strategy_key"], "symbol": t["symbol"],
                            "pnl_dollars": pnl_dollars, "r_multiple": r_multiple, "exit_time": ts})
        elif is_entry:
            equity = equity_now()
            peak_equity = max(peak_equity, equity)
            today = ts.date()
            if daily_start_date != today:
                daily_start_date = today
                daily_start_equity = equity

            drawdown_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            daily_loss_pct = (daily_start_equity - equity) / daily_start_equity if daily_start_equity > 0 else 0.0

            if drawdown_pct >= CIRCUIT_BREAKER_DD_PCT:
                entries_blocked_circuit += 1
                equity_curve.append((ts, equity_now()))
                continue
            if daily_loss_pct >= DAILY_LOSS_LIMIT_PCT:
                entries_blocked_daily += 1
                equity_curve.append((ts, equity_now()))
                continue

            # risk throttle, recomputed off trades closed so far
            log = closed
            three_losses = len(log) >= RISK_THROTTLE_CONSEC_LOSSES and all(
                c["r_multiple"] <= 0 for c in log[-RISK_THROTTLE_CONSEC_LOSSES:]
            )
            neg_expectancy = len(log) >= RISK_THROTTLE_LOOKBACK and (
                sum(c["r_multiple"] for c in log[-RISK_THROTTLE_LOOKBACK:]) / RISK_THROTTLE_LOOKBACK < 0
            )
            if three_losses or neg_expectancy:
                risk_throttled = True
            if equity >= peak_equity:
                risk_throttled = False
            effective_risk_pct = risk_pct * RISK_THROTTLE_MULT if risk_throttled else risk_pct

            risk_amount = equity * effective_risk_pct
            stop_distance = t["stop_distance"]
            if stop_distance is None or stop_distance <= 0:
                equity_curve.append((ts, equity_now()))
                continue
            notional = min((risk_amount / stop_distance) * t["entry_price"], cash)
            if notional <= 0:
                equity_curve.append((ts, equity_now()))
                continue
            cash -= notional
            open_positions[tid] = {"notional": notional, "entry_price": t["entry_price"],
                                    "pnl_pct": t["pnl_pct"], "strategy_key": t["strategy_key"],
                                    "risk_amount": risk_amount}
            entries_taken += 1

        equity_curve.append((ts, equity_now()))

    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    eq_df = eq_df[~eq_df.index.duplicated(keep="last")].sort_index()
    return closed, eq_df, entries_taken, entries_blocked_circuit, entries_blocked_daily


def summarize_replay(risk_pct, closed, eq_df, entries_taken, blocked_circuit, blocked_daily):
    if eq_df.empty:
        return None
    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity / STARTING_EQUITY - 1) * 100
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr = ((final_equity / STARTING_EQUITY) ** (1 / years) - 1) * 100 if years > 0 and final_equity > 0 else float("nan")
    running_max = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - running_max) / running_max).min() * 100
    daily_eq = eq_df["equity"].resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0.0
    n = len(closed)
    wins = sum(1 for c in closed if c["pnl_dollars"] > 0)
    win_rate = wins / n if n else float("nan")
    trades_per_year = n / years if years > 0 else float("nan")
    min_equity = eq_df["equity"].min()

    return {
        "risk_pct": risk_pct, "final_equity": final_equity, "total_return": total_return,
        "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "n_trades": n, "win_rate": win_rate * 100,
        "trades_per_year": trades_per_year, "blocked_circuit": blocked_circuit,
        "blocked_daily": blocked_daily, "min_equity": min_equity, "years": years,
    }


# --------------------------------------------------------------------------
# Layer 2: trade-sequence Monte Carlo (order-shuffle, fixed-fractional
# compounding, no concurrency) for risk-of-ruin estimation.
# --------------------------------------------------------------------------

def monte_carlo_ruin(r_multiples, risk_pct, n_sims=N_MONTE_CARLO, seed=42):
    rng = np.random.default_rng(seed)
    r = np.asarray(r_multiples, dtype=float)
    n = len(r)
    final_returns = np.empty(n_sims)
    max_dds = np.empty(n_sims)

    for i in range(n_sims):
        order = rng.permutation(n)
        seq = r[order]
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        consec_losses = 0
        window = []
        throttled = False
        for ri in seq:
            eff_risk = risk_pct * RISK_THROTTLE_MULT if throttled else risk_pct
            equity *= (1 + eff_risk * ri)
            equity = max(equity, 1e-9)  # fixed-fractional can't literally hit exactly 0
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

            window.append(ri)
            window = window[-RISK_THROTTLE_LOOKBACK:]
            consec_losses = consec_losses + 1 if ri <= 0 else 0
            three_losses = consec_losses >= RISK_THROTTLE_CONSEC_LOSSES
            neg_expectancy = len(window) >= RISK_THROTTLE_LOOKBACK and sum(window) / len(window) < 0
            if three_losses or neg_expectancy:
                throttled = True
            if equity >= peak:
                throttled = False

        final_returns[i] = equity - 1.0
        max_dds[i] = max_dd

    return final_returns, max_dds


def summarize_monte_carlo(risk_pct, final_returns, max_dds):
    out = {
        "risk_pct": risk_pct,
        "median_return": np.median(final_returns) * 100,
        "p05_return": np.percentile(final_returns, 5) * 100,
        "p95_return": np.percentile(final_returns, 95) * 100,
        "median_max_dd": np.median(max_dds) * 100,
        "worst5pct_max_dd": np.percentile(max_dds, 95) * 100,
        "p_net_loss": float(np.mean(final_returns < 0)) * 100,
    }
    for thr in RUIN_THRESHOLDS:
        out[f"p_dd_ge_{int(thr*100)}"] = float(np.mean(max_dds >= thr)) * 100
    return out


def main():
    all_trades = build_trades()
    if not all_trades:
        print("No trades generated -- aborting.")
        return

    print(f"{'='*100}")
    print("LAYER 1: HISTORICAL SHARED-LEDGER REPLAY (real concurrency, real risk management, one path)")
    print(f"{'='*100}")
    replay_rows = []
    closed_at_1pct = None
    for rp in RISK_LEVELS:
        closed, eq_df, taken, b_circuit, b_daily = replay_shared_ledger(all_trades, rp)
        stats = summarize_replay(rp, closed, eq_df, taken, b_circuit, b_daily)
        if stats:
            replay_rows.append(stats)
        if abs(rp - 0.01) < 1e-9:
            closed_at_1pct = closed

    hdr = (f"{'risk%':>7}{'trades':>8}{'trades/yr':>11}{'win%':>7}{'return%':>10}"
           f"{'CAGR%':>9}{'maxDD%':>9}{'Sharpe':>8}{'min_eq$':>11}{'blk_circ':>9}{'blk_day':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in replay_rows:
        print(f"{s['risk_pct']*100:>6.2f}%{s['n_trades']:>8}{s['trades_per_year']:>11.1f}"
              f"{s['win_rate']:>6.1f}%{s['total_return']:>9.1f}%{s['cagr']:>8.1f}%"
              f"{s['max_dd']:>8.1f}%{s['sharpe']:>8.2f}{s['min_equity']:>11,.0f}"
              f"{s['blocked_circuit']:>9}{s['blocked_daily']:>8}")

    print(f"\n{'='*100}")
    print(f"LAYER 2: TRADE-SEQUENCE MONTE CARLO ({N_MONTE_CARLO} reshuffles/level, no concurrency -- see caveats)")
    print(f"{'='*100}")
    r_multiples = [c["r_multiple"] for c in closed_at_1pct]
    print(f"Source: {len(r_multiples)} realized trade R-multiples from the 1%-risk historical replay\n")

    mc_hdr = (f"{'risk%':>7}{'median_ret%':>13}{'p05_ret%':>10}{'p95_ret%':>10}"
              f"{'med_maxDD%':>12}{'p95_maxDD%':>11}{'P(net loss)':>13}"
              + "".join(f"{'P(DD>=' + str(int(t*100)) + '%)':>14}" for t in RUIN_THRESHOLDS))
    print(mc_hdr)
    print("-" * len(mc_hdr))
    mc_rows = []
    for rp in RISK_LEVELS:
        final_returns, max_dds = monte_carlo_ruin(r_multiples, rp)
        mc = summarize_monte_carlo(rp, final_returns, max_dds)
        mc_rows.append(mc)
        line = (f"{rp*100:>6.2f}%{mc['median_return']:>12.1f}%{mc['p05_return']:>9.1f}%"
                f"{mc['p95_return']:>9.1f}%{mc['median_max_dd']:>11.1f}%{mc['worst5pct_max_dd']:>10.1f}%"
                f"{mc['p_net_loss']:>12.1f}%")
        for thr in RUIN_THRESHOLDS:
            line += f"{mc[f'p_dd_ge_{int(thr*100)}']:>13.1f}%"
        print(line)

    print(f"\n{'='*100}")
    print("CAVEATS")
    print(f"{'='*100}")
    print("""
- Layer 1 is ONE historical path (2016-2026 Alpaca daily/4H data) -- real DD/return/trade-frequency
  numbers, but only one specific ordering of bull/bear/chop regimes actually occurred.
- Layer 2 ignores position concurrency/overlap entirely (it replays trades one-at-a-time in a
  shuffled sequence, not the real calendar) -- this UNDERSTATES true portfolio risk somewhat,
  since in reality several positions can be open and losing at once (no position cap live), which
  the sequential model can't represent. Treat Layer 2's ruin probabilities as a floor, not a ceiling.
- Both layers share the same known correlated-symbols caveat as every other basket backtest in this
  project (trade count overstates independent sample size -- many symbols move together).
- Costs/slippage: PCSE's engine includes a 0.05%/trade cost; breakout/swing_breakout's engines do not
  model costs explicitly (same as this project's other breakout backtests).
- "Chance of exploding" here means P(max drawdown >= threshold) under Layer 2's compounding model,
  not literal ruin (equity can asymptotically approach but not hit exactly $0 under fixed-fractional
  sizing) -- a -90% drawdown is the realistic proxy for "the account is effectively wiped out."
""")


if __name__ == "__main__":
    main()
