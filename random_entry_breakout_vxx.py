"""Random-entry benchmark for Breakout Hunter + VXX 'not_spiking' filter, on
the same 40-symbol universe as run_basket_breakout_vxx.py (lighter-weight
than random_entry_all_bots.py's point-in-time S&P pipeline, whose bar cache
isn't available in this checkout -- same question, smaller/faster universe).

For each symbol: take the REAL not_spiking entry count, then draw N_DRAWS
sets of that many random entry days (past warmup, valid ATR/support), run
each through the identical exit engine, and compare pooled stats to the
real entries on the same symbols.
"""
import numpy as np
import pandas as pd

import strategy_breakout as strat
import strategy_breakout_vxx as vxx_strat
from backtest_breakout_vxx import run_backtest, WARMUP_BARS
from basket import US_SYMBOLS, STARTING_EQUITY, RISK_PCT
from data import fetch_yfinance
from test_mtf_oos_symbols import OOS_US_SYMBOLS

START = "2019-01-01"
END = "2026-08-01"
VARIANT = "long_entry_not_spiking"
N_DRAWS = 10
RNG = np.random.default_rng(42)


def pooled_stats(trades):
    if not trades:
        return {"num_trades": 0, "avg_r": 0.0, "t_stat": 0.0}
    r = np.array([t.r_multiple() for t in trades])
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    return {"num_trades": len(trades), "avg_r": round(float(r.mean()), 3),
            "t_stat": round(float(r.mean() / se), 2) if se > 0 else 0.0}


def main():
    vxx = fetch_yfinance("VXX", START, END)
    symbols = US_SYMBOLS + OOS_US_SYMBOLS

    real_trades_all = []
    random_draws = [[] for _ in range(N_DRAWS)]

    for symbol in symbols:
        try:
            df = fetch_yfinance(symbol, START, END)
        except Exception as e:
            print(f"  skip {symbol}: {e}")
            continue

        real_trades, _ = run_backtest(df, vxx, VARIANT, starting_equity=STARTING_EQUITY, risk_pct=RISK_PCT)
        real_trades = [t for t in real_trades if t.exit_price is not None]
        real_trades_all.extend(real_trades)
        n_real = len(real_trades)
        if n_real == 0:
            continue

        prepared = strat.prepare(df)
        data = vxx_strat.add_filters(prepared, vxx)
        rows = data.iloc[WARMUP_BARS:]
        valid_idx = rows.index[rows["support"].notna() & rows["atr"].notna()]
        if len(valid_idx) < n_real:
            continue

        for draw in range(N_DRAWS):
            chosen = RNG.choice(len(valid_idx), size=n_real, replace=False)
            fake_col = pd.Series(False, index=data.index)
            fake_col.loc[valid_idx[chosen]] = True
            data_copy = data.copy()
            data_copy["_random_entry"] = fake_col

            # reuse run_backtest's loop by monkey-patching the entry col via a thin wrapper
            from backtest_breakout_vxx import Trade
            cash, position, pending_entry, trades = STARTING_EQUITY, None, None, []
            for date, row in data_copy.iloc[WARMUP_BARS:].iterrows():
                if position is None and pending_entry is not None:
                    support, atr_at_signal = pending_entry
                    entry_price = row["Open"]
                    initial_stop = support - strat.INITIAL_STOP_ATR_BUFFER * atr_at_signal
                    emergency_stop = initial_stop - strat.EMERGENCY_STOP_BUFFER_ATR * atr_at_signal
                    stop_distance = entry_price - initial_stop
                    if stop_distance > 0:
                        risk_amount = cash * RISK_PCT
                        shares = min(risk_amount / stop_distance, cash / entry_price)
                        if shares > 0:
                            position = Trade(side="long", entry_date=date, entry_price=entry_price, shares=shares,
                                              initial_stop=initial_stop, emergency_stop=emergency_stop,
                                              atr_at_entry=atr_at_signal, risk_amount=risk_amount)
                            position.trail_active, position.trail_stop, position.extreme = False, None, entry_price
                            cash -= shares * entry_price
                    pending_entry = None

                if position is not None:
                    position.extreme = max(position.extreme, row["High"])
                    favorable = row["Close"] - position.entry_price
                    if not position.trail_active and favorable >= strat.TRAIL_ACTIVATE_ATR * position.atr_at_entry:
                        position.trail_active = True
                    if position.trail_active:
                        candidate = position.extreme - strat.TRAIL_ATR_MULT * row["atr"]
                        position.trail_stop = candidate if position.trail_stop is None else max(position.trail_stop, candidate)
                    active_stop = position.trail_stop if position.trail_active else position.initial_stop
                    fill_price = None
                    if row["Low"] <= position.emergency_stop:
                        fill_price = row["Open"] if row["Open"] <= position.emergency_stop else position.emergency_stop
                    elif row["Low"] <= active_stop:
                        fill_price = row["Open"] if row["Open"] <= active_stop else active_stop
                    if fill_price is not None:
                        position.exit_date, position.exit_price = date, fill_price
                        cash += position.shares * fill_price
                        trades.append(position)
                        position = None

                if position is None and pending_entry is None and bool(row["_random_entry"]):
                    pending_entry = (row["support"], row["atr"])

            random_draws[draw].extend([t for t in trades if t.exit_price is not None])
        print(f"  done {symbol}: real_trades={n_real}")

    real_stats = pooled_stats(real_trades_all)
    print(f"\n=== REAL entries (not_spiking), n={len(symbols)} symbols ===")
    print(real_stats)

    draw_stats = [pooled_stats(d) for d in random_draws]
    avg_rs = [s["avg_r"] for s in draw_stats]
    beats = sum(1 for a in avg_rs if a >= real_stats["avg_r"])
    print(f"\n=== RANDOM entries, {N_DRAWS} draws ===")
    for i, s in enumerate(draw_stats):
        print(f"  draw {i}: {s}")
    print(f"\nRandom avg_R range: {min(avg_rs):.3f} .. {max(avg_rs):.3f}, mean {np.mean(avg_rs):.3f}")
    print(f"Random draws with avg_R >= real's avg_R ({real_stats['avg_r']}): {beats}/{N_DRAWS}")


if __name__ == "__main__":
    main()
