"""Shadow test: bullish-engulfing (daily) + ATR1.5 stop + 3xATR trailing exit.

Read-only. Fetches daily bars through broker_alpaca_v2's data client and records the
trades this rule would have taken in a JSON state file. Never places or cancels orders.

Rules match validate_bullish_engulfing_trail.py: signal on a completed daily bar,
entry at the next session's open, stop = entry - 1.5*ATR14(signal bar), trailing stop
= highest high since entry - 3*ATR14(signal bar) (never below the initial stop),
exits checked on each daily bar, stop-first with gap-through fills at the open.
One virtual position per symbol. Each run replays only bars it has not processed yet,
so reruns are idempotent.

Entries on or after SHADOW_START are the live out-of-sample record.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import broker_alpaca_v2 as broker
from shadow_universe import SHADOW_SYMBOLS

STATE_DIR = Path(os.getenv("SHADOW_STATE_DIR", Path(__file__).resolve().parent / "shadow_state"))
STATE_FILE = STATE_DIR / "shadow_state.json"
TRADES_FILE = STATE_DIR / "shadow_trades.csv"
SHADOW_START = pd.Timestamp(os.getenv("SHADOW_START", str(datetime.now(timezone.utc).date())), tz="UTC")
STOP_ATR, TRAIL_ATR = 1.5, 3.0
LOOKBACK_DAYS = 400


def atr14(df):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean()


def is_bullish_engulfing(df, i):
    o, c = df["Open"].values, df["Close"].values
    return (c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1])


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"positions": {}, "pending": {}, "last_date": {}}


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1, default=str))


def log_trade(row):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    header = not TRADES_FILE.exists()
    pd.DataFrame([row]).to_csv(TRADES_FILE, mode="a", header=header, index=False)


def _apply_bar(sym, pos, o, h, l, d, events):
    stop_cur = max(pos["stop0"], pos["hw"] - TRAIL_ATR * pos["atr"])
    if l <= stop_cur:
        fill = min(o, stop_cur)
        exit_net = fill * (1 - 0.0005)
        R = (exit_net - pos["entry"]) / (pos["entry"] - pos["stop0"])
        row = {"symbol": sym, "entry_date": pos["entry_date"], "exit_date": d,
               "entry": round(pos["entry"], 4), "exit": round(exit_net, 4), "stop0": round(pos["stop0"], 4),
               "R": round(R, 4), "ret_pct": round(100 * (exit_net / pos["entry"] - 1), 3)}
        log_trade(row)
        events.append(("exit", row))
        return True, None
    pos["hw"] = max(pos["hw"], h)
    return False, pos


def process_symbol(sym, df, state, today_utc_date):
    atr = atr14(df).values
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    last_done = state["last_date"].get(sym)
    pos = state["positions"].get(sym)
    pend = state["pending"].get(sym)
    events = []
    for i in range(len(df)):
        d = dates[i]
        if last_done and d <= last_done:
            continue
        if today_utc_date and pd.Timestamp(d).date() > today_utc_date:
            break
        o, h, l, c = (float(df[x].values[i]) for x in ("Open", "High", "Low", "Close"))
        exited_now = False
        if pos:
            exited_now, pos = _apply_bar(sym, pos, o, h, l, d, events)
        if pend and pos is None and not exited_now:
            e = o * (1 + 0.0005)
            stop0 = o - STOP_ATR * pend["atr"]
            if e - stop0 > 0:
                pos = {"entry": e, "entry_date": d, "stop0": stop0, "atr": pend["atr"], "hw": e}
                events.append(("enter", {"symbol": sym, "date": d, "entry": round(e, 4), "stop0": round(stop0, 4)}))
                _, pos = _apply_bar(sym, pos, o, h, l, d, events)
        pend = None
        if pos is None and i >= 1 and not np.isnan(atr[i]) and is_bullish_engulfing(df, i):
            pend = {"atr": float(atr[i]), "signal_date": d}
            events.append(("signal", {"symbol": sym, "date": d}))
        last_done = d
    state["positions"][sym] = pos if pos else None
    if pos is None:
        state["positions"].pop(sym, None)
    if pend:
        state["pending"][sym] = pend
    else:
        state["pending"].pop(sym, None)
    state["last_date"][sym] = last_done
    return events


def summary(state):
    rows = Path(TRADES_FILE)
    lines = []
    if rows.exists():
        t = pd.read_csv(rows)
        t["entry_date"] = pd.to_datetime(t["entry_date"], utc=True)
        live = t[t["entry_date"] >= SHADOW_START]
        for label, sub in [("all replayed", t), (f"since {SHADOW_START.date()} (live shadow)", live)]:
            if len(sub):
                lines.append(f"{label}: n={len(sub)} win%={100*(sub.R>0).mean():.1f} "
                             f"exp_R={sub.R.mean():.3f} mean_ret%={sub.ret_pct.mean():.2f}")
            else:
                lines.append(f"{label}: n=0")
    lines.append(f"open virtual positions: {sorted(k for k, v in state['positions'].items() if v)}")
    lines.append(f"pending signals (enter next session): {sorted(state['pending'])}")
    return lines


def main(dry_run=False):
    print(f"=== shadow_engulfing run {datetime.now(timezone.utc).isoformat()} (no orders) ===", flush=True)
    state = load_state()
    bars = broker.fetch_daily_bars(SHADOW_SYMBOLS, lookback_days=LOOKBACK_DAYS)
    today = datetime.now(timezone.utc).date()
    new_events = []
    for sym in SHADOW_SYMBOLS:
        df = bars.get(sym)
        if df is None or len(df) < 30:
            continue
        df = df.sort_index()
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC") if df.index.tz else pd.DatetimeIndex(df.index).tz_localize("UTC")
        for kind, row in process_symbol(sym, df, state, today):
            new_events.append((kind, row))
    if not dry_run:
        save_state(state)
    for kind, row in new_events:
        if kind != "signal":
            print(f"  {kind}: {row}", flush=True)
    print(f"bars fetched for {len(bars)}/{len(SHADOW_SYMBOLS)} symbols; "
          f"signals this run: {sum(1 for k, _ in new_events if k == 'signal')}", flush=True)
    for line in summary(state):
        print(line, flush=True)


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
