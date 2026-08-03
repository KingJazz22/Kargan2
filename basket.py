"""Shared basket-backtest logic: run the strategy across many symbols, pool the
trades, and report significance stats -- reused across timeframes and strategies."""
import numpy as np
import pandas as pd

from backtest import run_backtest as run_backtest_trend

STARTING_EQUITY = 100_000.0
RISK_PCT = 0.01

US_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "XOM", "UNH", "HD", "V", "JNJ", "WMT", "DIS", "NFLX", "AMD", "CRM",
]
CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
]


def run_one(symbol, fetch_fn, backtest_fn, backtest_kwargs):
    df = fetch_fn(symbol)
    trades, equity_df = backtest_fn(df, starting_equity=STARTING_EQUITY, risk_pct=RISK_PCT, **backtest_kwargs)
    return [t for t in trades if t.exit_price is not None], equity_df


def run_basket(fetch_fn, symbols_by_group=None, backtest_fn=run_backtest_trend, backtest_kwargs=None, verbose=True):
    if symbols_by_group is None:
        symbols_by_group = (("US", US_SYMBOLS), ("Crypto", CRYPTO_SYMBOLS))
    backtest_kwargs = backtest_kwargs or {}

    all_trades = []
    per_symbol_rows = []

    def vprint(*a, **kw):
        if verbose:
            print(*a, **kw)

    for group_name, symbols in symbols_by_group:
        for symbol in symbols:
            try:
                trades, equity_df = run_one(symbol, fetch_fn, backtest_fn, backtest_kwargs)
            except Exception as e:
                vprint(f"  skip {symbol}: {e}")
                continue

            if len(equity_df) == 0:
                vprint(f"  skip {symbol}: no bars after warmup")
                continue

            final_equity = equity_df["equity"].iloc[-1]
            ret_pct = (final_equity / STARTING_EQUITY - 1) * 100
            span = f"{equity_df.index[0].date()} -> {equity_df.index[-1].date()}"
            per_symbol_rows.append({
                "group": group_name, "symbol": symbol, "num_trades": len(trades),
                "return_pct": round(ret_pct, 2), "bars": len(equity_df),
            })
            for t in trades:
                t.symbol = symbol
                t.group = group_name
            all_trades.extend(trades)
            vprint(f"  {group_name:6s} {symbol:9s} trades={len(trades):3d}  return={ret_pct:+.2f}%  bars={len(equity_df):5d}  ({span})")

    vprint(f"\nTotal pooled trades: {len(all_trades)}")
    by_group = pd.DataFrame(per_symbol_rows)
    if not all_trades:
        return all_trades, by_group, {"num_trades": 0}

    r_mults = np.array([t.r_multiple() for t in all_trades])
    pnls = np.array([t.pnl() for t in all_trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = 100 * len(wins) / len(all_trades)
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    se = r_mults.std(ddof=1) / np.sqrt(len(r_mults)) if len(r_mults) > 1 else 0.0
    t_stat = r_mults.mean() / se if se > 0 else 0.0

    stats = {
        "num_trades": len(all_trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_r_multiple": round(float(r_mults.mean()), 3),
        "median_r_multiple": round(float(np.median(r_mults)), 3),
        "stdev_r_multiple": round(float(r_mults.std()), 3),
        "profit_factor": round(profit_factor, 2),
        "t_stat_vs_zero": round(float(t_stat), 2),
    }

    vprint("\n=== Pooled trade statistics (all symbols, R-multiples) ===")
    for k, v in stats.items():
        vprint(f"  {k}: {v}")

    exit_reasons = pd.Series([t.exit_reason for t in all_trades]).value_counts()
    vprint("\n  exit reason breakdown:")
    vprint(exit_reasons.to_string())

    vprint("\n=== Per-symbol summary ===")
    vprint(by_group.to_string(index=False))

    vprint("\n=== By group ===")
    for g_name, _ in symbols_by_group:
        g_trades = [t for t in all_trades if t.group == g_name]
        if not g_trades:
            continue
        g_r = np.array([t.r_multiple() for t in g_trades])
        vprint(f"  {g_name}: n={len(g_trades)}  win_rate={round(100*np.mean(g_r>0),1)}%  avg_R={round(float(g_r.mean()),3)}")

    return all_trades, by_group, stats
