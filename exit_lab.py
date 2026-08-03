"""Exit-design search for PCSE, holding entries FIXED.

FINDINGS_CANDLE_PROB.md's permutation test showed PCSE's candle-state entries
beat random timing at the 100th percentile -- real signal -- but the original
adaptive-trailing-stop exit destroyed that edge (random entries into the same
exit engine did even worse than the real signal, not better). This module
freezes the validated long-only entry signal and searches exit design
instead: fixed TP/SL at different stdev multiples, TP at the first touch of a
moving average (EMA/rolling-VWAP, 50/100/150/200), and TP at RSI/Stochastic/
Bollinger/Keltner levels -- all using this repo's own `indicators.py`
(canned indicators are fine for EXITS; the brief only required entries to
come from pure candle statistics).

Two exit-rule "kinds", so the fill-timing convention matches the rest of the
project:
  - "price": a per-bar target PRICE (fixed, or dynamic like a moving average
    or band) checked INTRABAR via High >= target, same bar -- a resting limit
    order, same convention as this repo's trailing-stop fills.
  - "signal": a per-bar boolean (e.g. RSI >= 70) checked on the bar's CLOSE,
    exit queued for the NEXT bar's open -- lookahead-safe, same convention as
    entries and the original edge_decay exit.
SL is always a fixed price target set at entry (entry_price - sl_mult *
entry-time ret_stdev * entry_price), checked intrabar, and takes priority
over TP if both would trigger on the same bar (conservative).
"""
import numpy as np
import pandas as pd

import indicators as ind
import strategy_candle_prob as strat

WARMUP_BARS = strat.TRAIN_BARS + 5
MA_PERIODS = (50, 100, 150, 200)


def prepare_entries(df: pd.DataFrame, **entry_params) -> pd.DataFrame:
    """Freeze PCSE's validated long-only entry signal (the expensive walk-
    forward fit happens once here), then attach every indicator needed to
    test alternative exits against it."""
    entry_params = {"long_only": True, **entry_params}
    out = strat.prepare(df, **entry_params)

    close, high, low, volume = out["Close"], out["High"], out["Low"], out["Volume"]
    typical = (high + low + close) / 3.0
    for period in MA_PERIODS:
        out[f"ema_{period}"] = ind.ema(close, period)
        out[f"vwap_{period}"] = (typical * volume).rolling(period).sum() / volume.rolling(period).sum()

    out["rsi_14"] = ind.rsi(close, 14)
    out["stoch_k"], out["stoch_d"] = ind.stochastic(high, low, close, 14, 3, 3)
    out["bb_mid"] = close.rolling(20).mean()
    out["bb_std"] = close.rolling(20).std()
    out["kc_mid"] = ind.ema(close, 20)
    out["kc_atr"] = ind.atr(high, low, close, 10)
    return out


# ---- TP rule factories: each returns (kind, fn) ----
# "price" fn signature: (entry_price, ret_stdev_at_entry, row) -> target price or None
# "signal" fn signature: (row) -> bool

def tp_stdev(mult: float):
    return ("price", lambda entry_price, ret_stdev, row: entry_price + mult * ret_stdev * entry_price)


def tp_ma(col: str):
    def f(entry_price, ret_stdev, row):
        val = row[col]
        return float(val) if pd.notna(val) and val > entry_price else None
    return ("price", f)


def tp_bb(num_std: float):
    def f(entry_price, ret_stdev, row):
        if pd.isna(row["bb_mid"]) or pd.isna(row["bb_std"]):
            return None
        val = row["bb_mid"] + num_std * row["bb_std"]
        return float(val) if val > entry_price else None
    return ("price", f)


def tp_kc(mult: float):
    def f(entry_price, ret_stdev, row):
        if pd.isna(row["kc_mid"]) or pd.isna(row["kc_atr"]):
            return None
        val = row["kc_mid"] + mult * row["kc_atr"]
        return float(val) if val > entry_price else None
    return ("price", f)


def tp_rsi(level: float):
    return ("signal", lambda row: pd.notna(row["rsi_14"]) and row["rsi_14"] >= level)


def tp_stoch(level: float):
    return ("signal", lambda row: pd.notna(row["stoch_k"]) and row["stoch_k"] >= level)


def simulate_exit(rows: pd.DataFrame, sl_mult: float, tp_rule: tuple, starting_equity: float = 100_000.0,
                   risk_pct: float = 0.01, cost_per_trade_pct: float = 0.0005) -> tuple[list, pd.DataFrame]:
    tp_kind, tp_fn = tp_rule
    cash = starting_equity
    position = None
    pending_entry = None
    pending_exit_reason = None
    trades = []
    equity_curve = []

    for date, row in rows.iterrows():
        # 1. pending signal-exit from a prior bar's close -> fill at this bar's open
        if position is not None and pending_exit_reason is not None:
            exit_price = row["Open"] * (1 - cost_per_trade_pct)
            pnl = (exit_price - position["entry_price"]) * position["shares"]
            cash += position["shares"] * exit_price
            trades.append({**position, "exit_date": date, "exit_price": exit_price,
                            "exit_reason": pending_exit_reason, "pnl": pnl,
                            "r_multiple": pnl / position["risk_amount"]})
            position, pending_exit_reason = None, None

        # 2. pending entry -> fill at this bar's open
        if position is None and pending_entry is not None:
            confidence, ret_stdev_at_signal = pending_entry
            entry_price = row["Open"] * (1 + cost_per_trade_pct)
            sl_distance = sl_mult * ret_stdev_at_signal * entry_price
            if sl_distance > 0 and not np.isnan(sl_distance):
                risk_amount = cash * risk_pct
                shares = min(risk_amount / sl_distance, cash / entry_price)
                if shares > 0:
                    position = {
                        "entry_date": date, "entry_price": entry_price, "shares": shares,
                        "confidence": confidence, "sl_price": entry_price - sl_distance,
                        "ret_stdev_at_entry": ret_stdev_at_signal, "risk_amount": risk_amount,
                    }
                    cash -= shares * entry_price
            pending_entry = None

        # 3. manage open position: SL first (conservative), then intrabar price-TP
        if position is not None:
            fill_price, reason = None, None
            if row["Low"] <= position["sl_price"]:
                fill_price, reason = position["sl_price"], "stop_loss"
            elif tp_kind == "price":
                target = tp_fn(position["entry_price"], position["ret_stdev_at_entry"], row)
                if target is not None and row["High"] >= target:
                    fill_price = row["Open"] if row["Open"] >= target else target
                    reason = "take_profit"

            if fill_price is not None:
                exit_price = fill_price * (1 - cost_per_trade_pct)
                pnl = (exit_price - position["entry_price"]) * position["shares"]
                cash += position["shares"] * exit_price
                trades.append({**position, "exit_date": date, "exit_price": exit_price,
                                "exit_reason": reason, "pnl": pnl,
                                "r_multiple": pnl / position["risk_amount"]})
                position = None

        # 4. signal-kind TP, checked on close -> queue for next bar's open
        if position is not None and tp_kind == "signal" and tp_fn(row):
            pending_exit_reason = "take_profit"

        # 5. entry signal -> queue for next bar's open
        if position is None and pending_entry is None and bool(row["long_entry"]):
            pending_entry = (row["long_confidence"], row["ret_stdev"])

        # 6. mark-to-market
        equity = cash + position["shares"] * row["Close"] if position is not None else cash
        equity_curve.append((date, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return trades, equity_df


def pooled_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "avg_r": 0.0, "t_stat": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    r = np.array([t["r_multiple"] for t in trades])
    pnl = np.array([t["pnl"] for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    t_stat = float(r.mean() / se) if se > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    return {
        "n": len(r), "avg_r": float(r.mean()), "t_stat": t_stat,
        "win_rate": float((r > 0).mean()), "profit_factor": pf,
    }


def run_pooled(dfs_prepared: dict, sl_mult: float, tp_rule: tuple, **sim_kwargs) -> dict:
    """dfs_prepared: {symbol: prepared_df (from prepare_entries)}. Pools trades
    across all symbols and returns one set of stats -- same convention as
    basket.run_basket."""
    all_trades = []
    for symbol, prep in dfs_prepared.items():
        rows = prep.iloc[WARMUP_BARS:]
        if len(rows) < 10:
            continue
        trades, _ = simulate_exit(rows, sl_mult, tp_rule, **sim_kwargs)
        all_trades.extend(trades)
    return pooled_stats(all_trades)
