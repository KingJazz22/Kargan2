"""Live/paper trading runner V3: ML flat-market long-short, on its OWN Alpaca
paper account (broker_alpaca_v3.py / state_store_v3.py) -- isolated from v1
and v2 so its equity curve can be compared directly against them.

What it trades (FINDINGS_ML_FLAT.md): once a week, among the 100-stock
universe, take the stocks in a flat regime (own ADX(14) < 25 on each of the
last 10 daily bars), score them with a Ridge model trained on all prior
history (target: next-5-day return vs SPY, ranked), go LONG the top 5 and
SHORT the bottom 5, equal dollars, and hold until the next week's rebalance.
Features, universe and model come from strategy_ml_flat.py -- the same code
backtest_ml_flat.py ran.

STATUS: a forward test, NOT a validated edge. It failed the pre-registered dev
bar (2019-22 t=-0.18; 2022 was -41bps/wk); the 2023-26 holdout was +31.9bps/wk
(t=2.28). Paper only, to find out which of those is the real behavior.

Deviations from the backtest, all minor:
  - the model is refit on every rebalance (backtest: once a year), on all rows
    whose forward return is fully known, with the same EMBARGO;
  - entry/exit at the first scheduled run (~9:35 ET) instead of the 9:30 open;
  - a short candidate that isn't easy-to-borrow is skipped for the next-lowest.
A SPY-below-200SMA pause was tested and NOT adopted (weeks in that state were
positive on average; pausing lowered holdout t from 2.28 to 2.02).

Risk: no per-position stops (none in the backtest). Account-level circuit
breaker: if equity falls CIRCUIT_BREAKER_DD_PCT below its peak, the book is
flattened and new rebalances halt until `halted` is cleared in
live_state_v3.json by hand.

Run cadence: railway_runner.py with RUNNER_TARGET=v3 (same 4 daily times as
v2). Only the first run of each trading week trades; the rest just report.

Usage:
    python live_trading_v3.py            # paper run, places real (paper) orders
    python live_trading_v3.py --dry-run  # compute and print the book only
"""
import argparse
from datetime import datetime

import pandas as pd

import broker_alpaca_v3 as broker
import strategy_ml_flat as mlf
from state_store_v3 import load_state, save_state

HISTORY_START = "2016-01-01"     # same history the walk-forward backtest trained on
GROSS_EXPOSURE = 1.0              # long + short notional as a fraction of equity (0.5 / 0.5)
MIN_FLAT_STOCKS = 10              # backtest skipped weeks with fewer flat stocks
MAX_SIGNAL_AGE_DAYS = 5           # refuse to trade on stale bars
CIRCUIT_BREAKER_DD_PCT = 0.15


def week_key(ts: pd.Timestamp) -> str:
    y, w, _ = ts.isocalendar()
    return f"{y}-W{w:02d}"


def compute_book(today_et: pd.Timestamp):
    """Returns (signal_date, longs, shorts, scored_df) or raises on bad data."""
    symbols = mlf.UNIVERSE + ["SPY"]
    bars = broker.fetch_daily_bars(symbols, HISTORY_START)
    if "SPY" not in bars:
        raise RuntimeError("no SPY bars")
    # drop today's still-forming bar: decide on the last COMPLETED day only,
    # like the backtest's "signal at the week's last close"
    cache = {}
    for s, df in bars.items():
        df = mlf.norm_index(df)
        cache[s] = df[df.index < today_et.normalize()]
    print(f"  bars: {len(cache)}/{len(symbols)} symbols, last completed day {cache['SPY'].index[-1].date()}")

    panel = mlf.build_panel(cache)
    features = mlf.feature_columns(panel)
    signal_date = panel["date"].max()
    if (today_et.normalize() - signal_date).days > MAX_SIGNAL_AGE_DAYS:
        raise RuntimeError(f"latest flat-stock bar {signal_date.date()} is stale")

    train = mlf.training_rows(panel, signal_date)
    model = mlf.make_ridge().fit(train[features], train["target"])
    scored = panel[panel["date"] == signal_date].copy()
    scored["score"] = model.predict(scored[features])
    print(f"  model: Ridge trained on {len(train)} flat stock-days through "
          f"{train['date'].max().date()}; {len(scored)} flat stocks on {signal_date.date()}")
    if len(scored) < MIN_FLAT_STOCKS:
        return signal_date, [], [], scored

    longs, _ = mlf.select_long_short(scored)
    shorts = []
    for sym in scored.sort_values("score")["symbol"]:
        if len(shorts) == mlf.TOP_K:
            break
        if sym in longs:
            continue
        if broker.is_shortable(sym):
            shorts.append(sym)
        else:
            print(f"  {sym}: not easy-to-borrow -- skipping as a short")
    return signal_date, longs, shorts, scored


def open_book(longs, shorts, equity, last_close, args):
    leg_notional = equity * GROSS_EXPOSURE / (2 * mlf.TOP_K)
    book = {}
    for side, syms in (("long", longs), ("short", shorts)):
        for sym in syms:
            if sym not in last_close:
                print(f"  {sym}: no recent price -- skipping")
                continue
            qty = int(leg_notional // last_close[sym])
            if qty < 1:
                continue
            print(f"  {side.upper():5s} {sym}: {qty} sh (~${qty * last_close[sym]:,.0f})")
            if args.dry_run:
                continue
            fill = broker.submit_market_order(sym, qty, "buy" if side == "long" else "sell")
            if not fill["filled"]:
                print(f"    {sym}: not filled -- {fill.get('error', 'timeout')}")
                continue
            book[sym] = {"side": side, "qty": fill["qty"], "entry_price": fill["fill_price"]}
    return book


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebalance", action="store_true",
                        help="rebalance even if this week was already traded")
    args = parser.parse_args()

    now_et = pd.Timestamp.now(tz="America/New_York")
    today_et = now_et.tz_localize(None)
    print(f"=== live_trading_v3 run: {datetime.now().isoformat()} {'(DRY RUN)' if args.dry_run else ''} ===")

    state = load_state()
    state.setdefault("weekly_log", [])
    equity = broker.get_equity()
    positions = broker.get_open_positions()
    peak = max(state.get("peak_equity", equity), equity)
    state["peak_equity"] = peak
    dd = (peak - equity) / peak if peak > 0 else 0.0
    unreal = sum(p["unrealized_pl"] for p in positions.values())
    print(f"Account equity: ${equity:,.2f} (peak ${peak:,.2f}, DD {dd:.1%}) | "
          f"{len(positions)} positions, unrealized ${unreal:+,.2f}")

    try:
        if not args.dry_run and not broker.market_is_open():
            print("market closed -- nothing to do")
            return

        if dd >= CIRCUIT_BREAKER_DD_PCT and not state.get("halted"):
            print(f"!!! CIRCUIT BREAKER: {dd:.1%} below peak -- flattening and halting rebalances")
            if not args.dry_run:
                broker.close_all_positions()
                state["halted"] = True
        if state.get("halted"):
            print("halted -- clear `halted` in live_state_v3.json to resume")
            return

        wk = week_key(today_et)
        if state.get("last_rebalance_week") == wk and not args.force_rebalance:
            print(f"already rebalanced for {wk} -- monitoring only")
            return

        print(f"\n--- weekly rebalance {wk} ---")
        signal_date, longs, shorts, scored = compute_book(today_et)
        print(f"  longs:  {longs}\n  shorts: {shorts}")

        if not args.dry_run:
            if positions:
                print(f"  closing last week's book ({len(positions)} positions)")
                if not broker.close_all_positions():
                    raise RuntimeError("positions still open after close_all_positions -- not opening a new book")
            equity_after_close = broker.get_equity()
            if state.get("book_equity_start"):
                week_ret = equity_after_close / state["book_equity_start"] - 1
                state["weekly_log"].append({"week": state.get("last_rebalance_week"),
                                            "equity_start": state["book_equity_start"],
                                            "equity_end": equity_after_close, "return": week_ret})
                print(f"  last week's book: {week_ret:+.2%}")
            equity = equity_after_close

        if not longs:
            print(f"  only {len(scored)} flat stocks (< {MIN_FLAT_STOCKS}) -- staying flat this week")
            book = {}
        else:
            book = open_book(longs, shorts, equity, _last_closes(longs + shorts), args)

        if not args.dry_run:
            state.update({"last_rebalance_week": wk, "signal_date": str(signal_date.date()),
                          "book": book, "book_equity_start": equity,
                          "scores": scored.set_index("symbol")["score"].round(4).to_dict()})
    except Exception as e:
        print(f"\n!!! run aborted by unexpected error: {e!r} -- saving state")
    finally:
        if not args.dry_run:
            save_state(state)
    print("\n=== done ===")


def _last_closes(symbols):
    bars = broker.fetch_daily_bars(symbols, (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
    return {s: float(df["Close"].iloc[-1]) for s, df in bars.items()}


if __name__ == "__main__":
    main()
