"""Re-tests FINDINGS_BREAKOUT_PCSE_EXIT.md's headline recommendation
(sl_atr_mult=8 for the Structure-Confirmed Breakout + PCSE-exit combo, picked
over higher-t-stat wider stops because sl=8 had the best REAL dollar return)
on the realistic, survivorship-bias-corrected 81-symbol universe from
sweep_risk_levels_realistic.py, side by side with the original 40/47-symbol
universe that finding was actually measured on.

Replays the SAME 3-strategy combo actually deployed in live_trading_v2.py --
Breakout Hunter + Structure-Confirmed Breakout w/ PCSE-exit (the sl_atr_mult
parameter under test) + PCSE -- sharing one capital pool, through the same
risk-management-aware shared ledger (circuit breaker/daily loss gate/risk
throttle, no cap) as sweep_risk_levels.py, at today's live RISK_PCT=0.02.

sl_atr_mult grid matches FINDINGS_BREAKOUT_PCSE_EXIT.md's own sweep:
2, 3, 5, 8, 12, 15, 18, 20, 25, 30.
"""
import backtest_breakout_bb_exit as bt_bb_exit
import strategy_swing_breakout as strat_swing
import sweep_risk_levels as base
import sweep_risk_levels_realistic as sr
from alpaca_fetch import fetch_stock_daily, fetch_stock_4h

SL_GRID = [2, 3, 5, 8, 12, 15, 18, 20, 25, 30]
BB_NUM_STD = 2.5              # matches live_trading_v2.py's SWING_PCSE_BB_NUM_STD
LIVE_RISK_PCT = 0.02          # today's actual live setting (just raised from 0.01)
COST_PER_TRADE_PCT = 0.0005   # per leg (0.05% each way = 0.10% round trip), matches PCSE's own convention
BREAKOUT_ROUND_TRIP_COST = 0.001  # breakout_hunter's engine has no built-in cost modeling -- applied post-hoc


def build_fixed_trades(symbols, pcse_symbols, label):
    """Breakout Hunter + PCSE trades -- independent of sl_atr_mult, built once per universe."""
    cache = sr.load_cache()
    trades = []

    print(f"--- [{label}] Breakout Hunter, {len(symbols)} symbols ---")
    for symbol in symbols:
        key = ("daily", symbol)
        if key not in cache:
            try:
                cache[key] = fetch_stock_daily(symbol)
            except Exception as e:
                print(f"  skip {symbol} daily: {e}")
                cache[key] = None
        df = cache[key]
        if df is None or len(df) <= base.bt_bo.WARMUP_BARS + 5:
            continue
        tr, _ = base.bt_bo.run_backtest(df, starting_equity=1e12, risk_pct=0.01)
        for t in tr:
            if t.exit_price is None:
                continue
            trades.append({
                "strategy_key": "breakout_hunter", "symbol": symbol,
                "entry_time": base.to_utc(t.entry_date), "exit_time": base.to_utc(t.exit_date),
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "pnl_pct": (t.exit_price / t.entry_price - 1) - BREAKOUT_ROUND_TRIP_COST,
                "stop_distance": t.entry_price - t.initial_stop,
            })
    print(f"    {sum(1 for t in trades if t['strategy_key']=='breakout_hunter')} breakout_hunter trades")

    print(f"--- [{label}] PCSE, {len(pcse_symbols)} symbols ---")
    n_pcse = 0
    for symbol in pcse_symbols:
        key = ("4h", symbol)
        if key not in cache:
            try:
                cache[key] = fetch_stock_4h(symbol)
            except Exception as e:
                print(f"  skip {symbol} 4h: {e}")
                cache[key] = None
        df = cache[key]
        if df is None or len(df) <= base.bt_pcse.WARMUP_BARS + 5:
            continue
        try:
            tr, _ = base.bt_pcse.run_backtest(df, starting_equity=1e12, risk_pct=0.01)
        except Exception as e:
            print(f"  skip {symbol} pcse: {e}")
            continue
        for t in tr:
            if t.exit_price is None:
                continue
            raw_pnl_pct = (t.exit_price / t.entry_price - 1) * (1 if t.side == "long" else -1)
            trades.append({
                "strategy_key": "pcse", "symbol": symbol,
                "entry_time": base.to_utc(t.entry_date), "exit_time": base.to_utc(t.exit_date),
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "pnl_pct": raw_pnl_pct, "stop_distance": t.stop_distance,
            })
        n_pcse += sum(1 for t in tr if t.exit_price is not None)
    print(f"    {n_pcse} pcse trades")

    sr.save_cache(cache)
    return trades


def build_swing_pcse_exit_trades(symbols, sl_atr_mult, cache):
    trades = []
    for symbol in symbols:
        df = cache.get(("daily", symbol))
        if df is None:
            continue
        try:
            tr, _ = bt_bb_exit.run_backtest(
                df, strat_swing, starting_equity=1e12, risk_pct=0.01,
                sl_atr_mult=sl_atr_mult, bb_num_std=BB_NUM_STD,
                cost_per_trade_pct=COST_PER_TRADE_PCT, warmup_bars=70,
            )
        except Exception:
            continue
        for t in tr:
            if t.exit_price is None:
                continue
            trades.append({
                "strategy_key": "swing_pcse_exit", "symbol": symbol,
                "entry_time": base.to_utc(t.entry_date), "exit_time": base.to_utc(t.exit_date),
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "pnl_pct": t.exit_price / t.entry_price - 1, "stop_distance": t.stop_distance,
            })
    return trades


def run_universe(label, symbols, pcse_symbols):
    print(f"\n{'#'*100}\n# UNIVERSE: {label} ({len(symbols)} breakout/swing symbols, {len(pcse_symbols)} PCSE symbols)\n{'#'*100}")
    fixed_trades = build_fixed_trades(symbols, pcse_symbols, label)
    cache = sr.load_cache()

    hdr = f"{'sl_atr_mult':>12}{'trades':>8}{'win%':>7}{'return%':>10}{'CAGR%':>9}{'maxDD%':>9}{'Sharpe':>8}{'blk_circ':>9}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for sl in SL_GRID:
        swing_trades = build_swing_pcse_exit_trades(symbols, sl, cache)
        all_trades = sorted(fixed_trades + swing_trades, key=lambda t: t["entry_time"])
        closed, eq_df, taken, b_circ, b_day = base.replay_shared_ledger(all_trades, LIVE_RISK_PCT)
        s = base.summarize_replay(sl, closed, eq_df, taken, b_circ, b_day)
        if s is None:
            continue
        rows.append((sl, s))
        n_swing = sum(1 for t in swing_trades if True)
        print(f"{sl:>12}{s['n_trades']:>8}{s['win_rate']:>6.1f}%{s['total_return']:>9.1f}%"
              f"{s['cagr']:>8.1f}%{s['max_dd']:>8.1f}%{s['sharpe']:>8.2f}{s['blocked_circuit']:>9}"
              f"   (swing_pcse_exit legs: {n_swing})")

    best = max(rows, key=lambda r: r[1]["total_return"]) if rows else None
    if best:
        print(f"\nBest total return in {label}: sl_atr_mult={best[0]} ({best[1]['total_return']:+.1f}%)")
    return rows


def main():
    print(f"Testing at LIVE_RISK_PCT={LIVE_RISK_PCT*100:.0f}% (today's deployed setting)")
    orig = run_universe("ORIGINAL (survivorship-biased, 40/20 symbols -- what FINDINGS_BREAKOUT_PCSE_EXIT.md used)",
                         sr.ORIGINAL_SYMBOLS, sr.ORIGINAL_PCSE_SYMBOLS)
    realistic = run_universe("REALISTIC (81/61 symbols, +41 laggards/turnarounds)",
                              sr.EXPANDED_SYMBOLS, sr.EXPANDED_PCSE_SYMBOLS)

    print(f"\n{'='*100}")
    print("SIDE-BY-SIDE: total return by sl_atr_mult")
    print(f"{'='*100}")
    print(f"{'sl_atr_mult':>12}{'original%':>12}{'realistic%':>13}")
    orig_by_sl = {sl: s["total_return"] for sl, s in orig}
    real_by_sl = {sl: s["total_return"] for sl, s in realistic}
    for sl in SL_GRID:
        o = orig_by_sl.get(sl)
        r = real_by_sl.get(sl)
        o_str = f"{o:+.1f}%" if o is not None else "n/a"
        r_str = f"{r:+.1f}%" if r is not None else "n/a"
        print(f"{sl:>12}{o_str:>12}{r_str:>13}")


if __name__ == "__main__":
    main()
