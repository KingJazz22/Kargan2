"""Full Kargan2 live-system simulation: Breakout Hunter (daily) + MTF v2
(4H+Daily, VWAP(200) trend filter) + PCSE (4H, SL=12x/TP=BB(3.0std)) sharing
ONE capital pool -- matches live_trading.py's actual current configuration:
same MAX_CONCURRENT_POSITIONS=10 cap across all three, same 1% risk sizing,
same symbol scoping (Breakout+MTF on the 40-symbol SYMBOLS universe, PCSE
restricted to the 20-symbol basket.US_SYMBOLS it was actually validated on).

Grid-Martingale is NOT included -- it isn't wired into live_trading.py (its
own FINDINGS_GRID_MARTINGALE.md says "still not ready for live paper trading").

Trade generation is decoupled from capital (each engine run with an
effectively unlimited starting equity), then replayed together through one
shared ledger that applies the REAL 1%-risk/stop-distance sizing and the
REAL shared 10-position cap -- same approach as simulate_full_shared_account.py.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import simulate_live_system as k2sim
import backtest_pcse_final as bt_pcse
from basket import US_SYMBOLS
from alpaca_fetch import fetch_stock_4h

STARTING_EQUITY = 9995.18  # same baseline used throughout this project's simulations
MAX_CONCURRENT_POSITIONS = 10  # across ALL THREE strategies, matching live_trading.py
RISK_PCT = 0.01
PCSE_SYMBOLS = US_SYMBOLS  # PCSE was only validated on this 20-symbol half of SYMBOLS


def to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def build_breakout_mtf_trades():
    print("=== Breakout Hunter + MTF v2 (VWAP(200) filter) ===")
    trades, _ = k2sim.run_combined(
        k2sim.SYMBOLS, starting_equity=1e12, risk_pct=RISK_PCT,
        max_concurrent_positions=None, mtf_trend_filter=("vwap", 200), verbose=True,
    )
    out = []
    for pos in trades:
        if pos.exit_price is None:
            continue
        strategy_key = "breakout_hunter" if pos.strategy == "breakout" else "mtf_v2"
        if pos.strategy == "breakout":
            stop_distance = pos.entry_price - pos.initial_stop
        else:
            stop_distance = k2sim.strat_mtf.TRAIL_ATR_MULT * pos.atr_at_entry
        out.append({
            "strategy_key": strategy_key, "symbol": pos.symbol,
            "entry_time": to_utc(pos.entry_date), "exit_time": to_utc(pos.exit_date),
            "entry_price": pos.entry_price, "exit_price": pos.exit_price,
            "pnl_pct": pos.exit_price / pos.entry_price - 1, "stop_distance": stop_distance,
        })
    print(f"  {len(out)} trades total (breakout + mtf)\n")
    return out


def build_pcse_trades():
    print("=== PCSE (4H, SL=12x / TP=BB(3.0std)) ===")
    out = []
    for symbol in PCSE_SYMBOLS:
        try:
            df = fetch_stock_4h(symbol)
        except Exception as e:
            print(f"  skip {symbol}: fetch failed: {e}")
            continue
        if len(df) <= bt_pcse.WARMUP_BARS + 5:
            print(f"  skip {symbol}: only {len(df)} bars (need > {bt_pcse.WARMUP_BARS + 5})")
            continue
        try:
            trades, _ = bt_pcse.run_backtest(df, starting_equity=1e12, risk_pct=RISK_PCT)
        except Exception as e:
            print(f"  skip {symbol}: backtest failed: {e}")
            continue
        for t in trades:
            if t.exit_price is None:
                continue
            pnl_pct = (t.exit_price / t.entry_price - 1) * (1 if t.side == "long" else -1)
            out.append({
                "strategy_key": "pcse", "symbol": symbol,
                "entry_time": to_utc(t.entry_date), "exit_time": to_utc(t.exit_date),
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "pnl_pct": pnl_pct, "stop_distance": t.stop_distance,
            })
        print(f"  {symbol}: {len(trades)} trades")
    print(f"  {len(out)} PCSE trades total\n")
    return out


def build_price_series(symbols):
    print("=== Fetching mark-to-market price series (Alpaca daily -- MUST match the")
    print("    price source trades were generated from, or split-adjustment scale")
    print("    mismatches corrupt unrealized PnL on any symbol that split since) ===")
    from alpaca_fetch import fetch_stock_daily
    series = {}
    for sym in symbols:
        try:
            df = fetch_stock_daily(sym)
            s = df["Close"].copy()
            s.index = s.index.map(to_utc)
            series[sym] = s
        except Exception as e:
            print(f"  skip price series {sym}: {e}")
    return series


def last_price(series, symbol, ts, fallback):
    s = series.get(symbol)
    if s is None or len(s) == 0:
        return fallback
    try:
        v = s.asof(ts)
        return float(v) if pd.notna(v) else fallback
    except Exception:
        return fallback


@dataclass
class OpenPos:
    strategy_key: str
    symbol: str
    notional: float
    entry_price: float
    pnl_pct: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp


def run_shared_ledger(all_trades, price_series, max_open=None):
    max_open = MAX_CONCURRENT_POSITIONS if max_open is None else max_open
    events = []
    by_id = {}
    for i, t in enumerate(all_trades):
        tid = f"t{i}"
        by_id[tid] = t
        events.append((t["entry_time"], 1, tid))
        events.append((t["exit_time"], 0, tid))
    events.sort(key=lambda e: (e[0], e[1]))

    cash = STARTING_EQUITY
    open_positions: dict[str, OpenPos] = {}
    equity_curve = []
    closed = []
    accepted = []  # every position actually opened -- (strategy_key, symbol, entry_time, exit_time, notional, entry_price, pnl_pct)
    blocked_max_open = 0

    def equity_now(ts):
        mtm = sum(p.notional * (1 + (last_price(price_series, p.symbol, ts, p.entry_price) / p.entry_price - 1))
                  for p in open_positions.values())
        return cash + mtm

    for ts, _, tid in events:
        t = by_id[tid]
        is_entry = tid not in open_positions and t["entry_time"] == ts
        is_exit = tid in open_positions and t["exit_time"] == ts

        if is_exit:
            pos = open_positions.pop(tid)
            pnl_dollars = pos.notional * pos.pnl_pct
            cash += pos.notional + pnl_dollars
            closed.append({"strategy_key": pos.strategy_key, "symbol": pos.symbol,
                            "pnl_dollars": pnl_dollars, "pnl_pct": pos.pnl_pct, "exit_time": ts})
        elif is_entry:
            if len(open_positions) >= max_open:
                blocked_max_open += 1
                continue
            equity = equity_now(ts)
            risk_amount = equity * RISK_PCT
            stop_distance = t["stop_distance"]
            if stop_distance is None or stop_distance <= 0:
                continue
            notional = min((risk_amount / stop_distance) * t["entry_price"], cash)
            if notional <= 0:
                continue
            cash -= notional
            pos = OpenPos(t["strategy_key"], t["symbol"], notional, t["entry_price"], t["pnl_pct"],
                          t["entry_time"], t["exit_time"])
            open_positions[tid] = pos
            accepted.append(pos)

        equity_curve.append((ts, equity_now(ts)))

    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return closed, eq_df, blocked_max_open, accepted


def analyze_drawdown(eq_df, accepted, price_series):
    """Finds the single worst peak-to-trough drawdown and attributes it to
    the positions that were open (or closed with a loss) during that window."""
    eq_df = eq_df[~eq_df.index.duplicated(keep="last")].sort_index()
    running_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - running_max) / running_max

    trough_ts = drawdown.idxmin()
    dd_pct = drawdown.loc[trough_ts] * 100
    peak_ts = eq_df.loc[:trough_ts, "equity"].idxmax()
    peak_equity = eq_df.loc[peak_ts, "equity"]
    trough_equity = eq_df.loc[trough_ts, "equity"]

    print(f"\n{'='*70}")
    print("WORST DRAWDOWN -- ATTRIBUTION")
    print(f"{'='*70}")
    print(f"Peak:   {peak_ts.date()}  equity=${peak_equity:,.2f}")
    print(f"Trough: {trough_ts.date()}  equity=${trough_equity:,.2f}")
    print(f"Drawdown: {dd_pct:.1f}%  |  Duration: {(trough_ts - peak_ts).days} days")

    # positions open at any point during [peak_ts, trough_ts]
    overlapping = [p for p in accepted if p.entry_time <= trough_ts and p.exit_time >= peak_ts]

    rows = []
    for p in overlapping:
        if p.exit_time <= trough_ts:
            # closed during the drawdown window -- use its actual realized pnl
            pnl_dollars = p.notional * p.pnl_pct
            status = "closed"
            as_of = p.exit_time
        else:
            # still open at the trough -- mark-to-market unrealized pnl
            px = last_price(price_series, p.symbol, trough_ts, p.entry_price)
            pnl_dollars = p.notional * (px / p.entry_price - 1)
            status = "open@trough"
            as_of = trough_ts
        rows.append({"strategy": p.strategy_key, "symbol": p.symbol, "status": status,
                      "entry_time": p.entry_time.date(), "as_of": as_of.date() if hasattr(as_of, "date") else as_of,
                      "notional": p.notional, "pnl_dollars": pnl_dollars})

    if not rows:
        print("No positions overlapped this window (drawdown may be from mark-to-market noise only).")
        return

    df = pd.DataFrame(rows).sort_values("pnl_dollars")
    print(f"\n{len(df)} positions overlapped the drawdown window. By strategy:")
    by_strat = df.groupby("strategy")["pnl_dollars"].agg(["count", "sum"]).sort_values("sum")
    for strat, row in by_strat.iterrows():
        print(f"  {strat:<18} n={int(row['count']):>3}  total_pnl=${row['sum']:>+10,.2f}")

    print(f"\nWorst 15 individual positions during this window:")
    print(f"{'strategy':<16}{'symbol':<8}{'status':<14}{'entry':<12}{'as_of':<12}{'notional':>10}{'pnl':>12}")
    for _, r in df.head(15).iterrows():
        print(f"{r['strategy']:<16}{r['symbol']:<8}{r['status']:<14}{str(r['entry_time']):<12}"
              f"{str(r['as_of']):<12}{r['notional']:>10,.0f}{r['pnl_dollars']:>+12,.2f}")


def summarize(closed, eq_df, blocked_max_open, max_open):
    print(f"\n{'='*70}")
    print("FULL KARGAN2 LIVE SYSTEM -- Breakout Hunter + MTF v2 + PCSE")
    print(f"{'='*70}")
    print(f"Starting equity: ${STARTING_EQUITY:,.2f} | Max concurrent positions: {max_open}")

    if eq_df.empty:
        print("No events processed.")
        return None

    eq_df = eq_df[~eq_df.index.duplicated(keep="last")].sort_index()
    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity / STARTING_EQUITY - 1) * 100
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr = ((final_equity / STARTING_EQUITY) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    running_max = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - running_max) / running_max).min() * 100
    daily_eq = eq_df["equity"].resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0.0

    print(f"\nWindow: {eq_df.index[0].date()} -> {eq_df.index[-1].date()} ({years:.1f} yrs)")
    print(f"Final equity: ${final_equity:,.2f}")
    print(f"Total return: {total_return:+.1f}%")
    print(f"CAGR: {cagr:+.1f}%")
    print(f"Max drawdown: {max_dd:.1f}%")
    print(f"Sharpe (daily, annualized): {sharpe:.2f}")
    print(f"Entries blocked by the {max_open}-position cap: {blocked_max_open}")

    stats = {"max_open": max_open, "total_return": total_return, "max_dd": max_dd, "sharpe": sharpe}

    if closed:
        n = len(closed)
        wins = sum(1 for c in closed if c["pnl_dollars"] > 0)
        total_pnl = sum(c["pnl_dollars"] for c in closed)
        print(f"\nClosed trades: {n} | Win rate: {wins/n:.1%} | Total realized PnL: ${total_pnl:+,.2f}")

        print(f"\n{'strategy':<18}{'trades':>8}{'win%':>8}{'avg pnl%':>10}{'PnL':>14}")
        print("-" * 58)
        from collections import defaultdict
        by_key = defaultdict(list)
        for c in closed:
            by_key[c["strategy_key"]].append(c)
        for key, trades in sorted(by_key.items(), key=lambda kv: -sum(c["pnl_dollars"] for c in kv[1])):
            wr = sum(1 for c in trades if c["pnl_dollars"] > 0) / len(trades)
            pnl = sum(c["pnl_dollars"] for c in trades)
            avg_pnl_pct = np.mean([c["pnl_pct"] for c in trades]) * 100
            print(f"{key:<18}{len(trades):>8}{wr*100:>7.1f}%{avg_pnl_pct:>9.2f}%{pnl:>+14,.2f}")

    return stats


if __name__ == "__main__":
    bo_mtf_trades = build_breakout_mtf_trades()
    pcse_trades = build_pcse_trades()
    all_trades = bo_mtf_trades + pcse_trades

    all_symbols = sorted(set(t["symbol"] for t in all_trades))
    price_series = build_price_series(all_symbols)

    closed, eq_df, blocked, accepted = run_shared_ledger(all_trades, price_series, max_open=MAX_CONCURRENT_POSITIONS)
    summarize(closed, eq_df, blocked, MAX_CONCURRENT_POSITIONS)
    analyze_drawdown(eq_df, accepted, price_series)

    print(f"\n{'='*70}")
    print("POSITION-CAP SENSITIVITY SWEEP (same trades, replayed with different caps)")
    print(f"{'='*70}")
    print(f"{'cap':>5}{'return':>12}{'max_dd':>10}{'sharpe':>9}")
    for cap in [3, 4, 5, 6, 7, 8, 10]:
        c, e, b, a = run_shared_ledger(all_trades, price_series, max_open=cap)
        if e.empty:
            continue
        e = e[~e.index.duplicated(keep="last")].sort_index()
        final_equity = e["equity"].iloc[-1]
        total_return = (final_equity / STARTING_EQUITY - 1) * 100
        running_max = e["equity"].cummax()
        max_dd = ((e["equity"] - running_max) / running_max).min() * 100
        daily_eq = e["equity"].resample("1D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0.0
        print(f"{cap:>5}{total_return:>11.1f}%{max_dd:>9.1f}%{sharpe:>9.2f}")
