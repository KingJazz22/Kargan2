"""Simulates the FULL shared-account picture: Kargan2's live_trading.py
(Breakout Hunter + MTF v2) PLUS all 7 strategies actually running in the
`New Trading bot` project's trading-worker service -- both systems trade the
SAME real Alpaca paper account ("Kargan 2", key PKQG7S5YMWLW4YLFPPRGCSDP75),
each unaware of the other's positions or capital usage.

Reuses the ACTUAL strategy classes from New Trading bot (core/strategies/*.py)
via its own core.backtest.engine.run_backtest -- not a re-derivation of their
rules -- and Kargan2's own strategy_breakout.py/strategy_mtf.py via
simulate_live_system.run_combined(). Trade decisions (entry/exit timing and
price) are generated independent of capital, then replayed through ONE shared
ledger that applies each system's REAL position-sizing and risk logic:

  - Kargan2: fixed 1% equity risk / stop-distance sizing, one shared 10-position
    cap across both its strategies (matches live_trading.py exactly).
  - New Trading bot: half-Kelly sizing bootstrapped from each strategy's own
    realized win/loss history (falls back to a flat 2% of equity with no
    history, per core/sizing/kelly.py + scheduler/jobs.py), a Monte Carlo
    pre-trade gate that halves or blocks size based on that strategy's own
    trailing 50-trade P&L distribution, a portfolio circuit breaker (15%
    drawdown from peak -- watching the REAL SHARED account equity, so a loss
    caused by Kargan2's strategies can trip it), a daily-loss gate (4% of
    day-start equity, but only counting New Trading bot's OWN realized P&L,
    matching how core/risk/daily_loss.py is actually fed), and its own
    separate 10-position cap.

Simplifications (all in the direction of NOT overstating returns):
  - No ML gate effect modeled (no trained model artifact exists locally --
    core/strategies/base.py's check_ml_gate() fails open/passes-through with
    no model loaded, which is what an unTrained deploy actually does too).
  - No PDT day-trade-count gating (these are swing/overnight-hold strategies;
    modeling exact same-day round-trips across two independent systems
    sharing one account is out of scope here).
  - Mark-to-market between events uses each symbol's own price series via
    asof lookup, not tick-by-tick.
  - New Trading bot's live scan/execute split (queue at one job, execute at
    a later cron) collapses to same-bar-close entries, matching how New
    Trading bot's OWN validate_strategies.py already backtests it via
    core.backtest.engine.run_backtest.
"""
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

import alpaca_fetch as af

NEWBOT_ROOT = r"C:\Users\PC\Desktop\New Trading bot"
sys.path.insert(0, NEWBOT_ROOT)

from core.data.indicators import compute_all
from core.backtest.engine import run_backtest as newbot_run_backtest
from core.strategies.crypto_momentum import CryptoMomentum
from core.strategies.crypto_mean_rev import CryptoMeanReversion
from core.strategies.breakout import Breakout as NewbotBreakout
from core.strategies.bb_vwap_reversion import BBVwapReversion
from core.strategies.ema_cross import EmaCross
from core.strategies.equity_swing import EquitySwing
from core.sizing.kelly import half_kelly_fraction
from core.sizing.mc_gate import MonteCarloGate
from core.risk.circuit_breaker import CircuitBreaker
from core.risk.daily_loss import DailyLossGate

sys.path.insert(0, r"C:\Users\PC\Desktop\Kargan2")
import simulate_live_system as k2sim

# ── Config confirmed from Railway (trading-worker service) / config.py defaults ──
STARTING_EQUITY = 9995.18  # real current "Kargan 2" account equity

NEWBOT_CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD"]
LDO_YF = "LDO-USD"  # Alpaca only has LDO/USD history since 2026-02 -- Yahoo goes back further
BAT_ALPACA = "BAT/USD"
NEWBOT_EQUITY = ["QQQ", "SPY", "AAPL", "NVDA", "META", "MSFT", "AMD", "TSLA", "GLD", "XLK", "XLE"]
BREAKOUT_EQUITY_SYMBOLS = ["NKE", "CSCO", "NFLX", "WFC"]
BREAKOUT_EQUITY_PARAMS = dict(bb_squeeze_ratio=1.5, vol_surge_min=1.0, efficiency_min=0.15)

NEWBOT_RISK_PCT = 0.02
NEWBOT_MAX_OPEN = 10
NEWBOT_MAX_DD = 0.15
NEWBOT_DAILY_LOSS_LIMIT = 0.04
KARGAN2_MAX_OPEN = 10
WARMUP = 210  # >= ema_long_period(200) and REGIME_WINDOW_BARS(200)


# ── Data fetching. Equities + BTC/ETH/SOL/AVAX/LINK/BAT come from Alpaca
# (10yr equity / 5.6yr crypto history, vs yfinance's ~2yr hourly/4H cap).
# LDO stays on yfinance -- Alpaca only has LDO/USD since 2026-02. ──
def _lower(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_1h(symbol, is_crypto=False):
    df = af.fetch_crypto_1h(symbol) if is_crypto else af.fetch_stock_1h(symbol)
    return _lower(df)


def fetch_4h(symbol, is_crypto=False):
    df = af.fetch_crypto_4h(symbol) if is_crypto else af.fetch_stock_4h(symbol)
    return _lower(df)


def fetch_daily(symbol, is_crypto=False):
    df = af.fetch_crypto_daily(symbol) if is_crypto else af.fetch_stock_daily(symbol)
    return _lower(df)


def fetch_ldo_1h_yf():
    df = yf.download(LDO_YF, period="729d", interval="60m", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("no LDO data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


def to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ── Step 1: generate New Trading bot's raw trades via its OWN strategy classes ──
newbot_trades = []


def add_newbot(strategy_key, strategy_instance, symbol, raw_df, params=None):
    if len(raw_df) < WARMUP + 5:
        print(f"  skip {strategy_key}/{symbol}: only {len(raw_df)} bars")
        return
    prepared = compute_all(raw_df, params=params).dropna()
    if len(prepared) < WARMUP + 5:
        print(f"  skip {strategy_key}/{symbol}: only {len(prepared)} bars after indicator warmup")
        return
    trades = newbot_run_backtest(strategy_instance, prepared, symbol, warmup_bars=WARMUP)
    for t in trades:
        newbot_trades.append({
            "strategy_key": strategy_key, "symbol": symbol, "side": t.side,
            "entry_time": to_utc(t.entry_time), "exit_time": to_utc(t.exit_time),
            "entry_price": t.entry_price, "exit_price": t.exit_price, "pnl_pct": t.pnl_pct,
        })
    print(f"  {strategy_key}/{symbol}: {len(trades)} trades")


def build_newbot_trades():
    print("\n=== New Trading bot strategies (reusing their real strategy classes) ===")
    for ticker in NEWBOT_CRYPTO:
        try:
            df1h = fetch_1h(ticker, is_crypto=True)
        except Exception as e:
            print(f"  skip {ticker} 1h: {e}"); continue
        add_newbot("crypto_momentum", CryptoMomentum(), ticker, df1h)
        add_newbot("crypto_mean_rev", CryptoMeanReversion(), ticker, df1h)
        try:
            df4h = fetch_4h(ticker, is_crypto=True)
            add_newbot("breakout_crypto", NewbotBreakout(asset_class="crypto"), ticker, df4h)
        except Exception as e:
            print(f"  skip {ticker} 4h: {e}")

    try:
        df_ldo = fetch_ldo_1h_yf()
        add_newbot("bb_vwap_reversion_ldo", BBVwapReversion(band="bb"), LDO_YF, df_ldo,
                   params={"vwap_lookback": 7})
    except Exception as e:
        print(f"  skip LDO: {e}")

    try:
        df_bat = fetch_1h(BAT_ALPACA, is_crypto=True)
        add_newbot("ema_cross_bat", EmaCross(asset_class="crypto"), BAT_ALPACA, df_bat)
    except Exception as e:
        print(f"  skip BAT: {e}")

    for sym in NEWBOT_EQUITY:
        try:
            dfe = fetch_daily(sym)
            add_newbot("equity_swing", EquitySwing(), sym, dfe)
        except Exception as e:
            print(f"  skip {sym} daily: {e}")

    for sym in BREAKOUT_EQUITY_SYMBOLS:
        try:
            df4e = fetch_4h(sym)
            add_newbot("breakout_equity", NewbotBreakout(asset_class="equity", **BREAKOUT_EQUITY_PARAMS),
                       sym, df4e)
        except Exception as e:
            print(f"  skip {sym} 4h: {e}")


# ── Step 2: generate Kargan2's raw trades (decoupled from capital) ──
kargan2_trades = []


def build_kargan2_trades():
    print("\n=== Kargan2 (Breakout Hunter + MTF v2) ===")
    trades, _ = k2sim.run_combined(k2sim.SYMBOLS, starting_equity=1e12, risk_pct=0.01,
                                    max_concurrent_positions=None, verbose=False)
    for pos in trades:
        if pos.exit_price is None:
            continue
        strategy_key = "breakout_hunter" if pos.strategy == "breakout" else "mtf_v2"
        if pos.strategy == "breakout":
            stop_distance = pos.entry_price - pos.initial_stop
        else:
            stop_distance = k2sim.strat_mtf.TRAIL_ATR_MULT * pos.atr_at_entry
        kargan2_trades.append({
            "strategy_key": strategy_key, "symbol": pos.symbol, "side": "buy",
            "entry_time": to_utc(pos.entry_date), "exit_time": to_utc(pos.exit_date),
            "entry_price": pos.entry_price, "exit_price": pos.exit_price,
            "pnl_pct": pos.exit_price / pos.entry_price - 1, "stop_distance": stop_distance,
        })
    print(f"  {len(kargan2_trades)} trades total (breakout + mtf)")


# ── Step 3: price series for continuous mark-to-market ──
price_series = {}


def build_price_series():
    print("\n=== Fetching mark-to-market price series ===")
    equity_syms = sorted(set(NEWBOT_EQUITY) | set(BREAKOUT_EQUITY_SYMBOLS) | set(k2sim.SYMBOLS))
    for sym in equity_syms:
        try:
            df = fetch_daily(sym)
            s = df["close"].copy()
            s.index = s.index.map(to_utc)
            price_series[sym] = s
        except Exception as e:
            print(f"  skip price series {sym}: {e}")
    for ticker in NEWBOT_CRYPTO + [BAT_ALPACA]:
        try:
            df = fetch_1h(ticker, is_crypto=True)
            s = df["close"].copy()
            s.index = s.index.map(to_utc)
            price_series[ticker] = s
        except Exception as e:
            print(f"  skip price series {ticker}: {e}")
    try:
        df_ldo = fetch_ldo_1h_yf()
        s = df_ldo["close"].copy()
        s.index = s.index.map(to_utc)
        price_series[LDO_YF] = s
    except Exception as e:
        print(f"  skip price series LDO: {e}")


def last_price(symbol, ts, fallback):
    s = price_series.get(symbol)
    if s is None or len(s) == 0:
        return fallback
    try:
        v = s.asof(ts)
        return float(v) if pd.notna(v) else fallback
    except Exception:
        return fallback


# ── Step 4: shared-ledger chronological replay ──
@dataclass
class OpenPos:
    family: str
    strategy_key: str
    symbol: str
    side: str
    notional: float
    entry_price: float
    exit_price: float
    pnl_pct: float


def run_shared_ledger():
    all_trades = []
    for i, t in enumerate(newbot_trades):
        all_trades.append({**t, "id": f"nb{i}", "family": "newbot"})
    for i, t in enumerate(kargan2_trades):
        all_trades.append({**t, "id": f"k2{i}", "family": "kargan2"})

    events = []
    by_id = {t["id"]: t for t in all_trades}
    for t in all_trades:
        events.append((t["entry_time"], 1, t["id"]))
        events.append((t["exit_time"], 0, t["id"]))
    events.sort(key=lambda e: (e[0], e[1]))

    cash = STARTING_EQUITY
    open_positions: dict[str, OpenPos] = {}
    newbot_open, kargan2_open = 0, 0
    newbot_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "avg_win": 100.0, "avg_loss": 100.0,
                                         "win_pnls": [], "loss_pnls": []})
    newbot_pnl_hist = defaultdict(list)
    cb = CircuitBreaker(max_drawdown_pct=NEWBOT_MAX_DD)
    dlg = DailyLossGate(max_daily_loss_pct=NEWBOT_DAILY_LOSS_LIMIT)
    mc_gate = MonteCarloGate()

    equity_curve = []
    closed = []
    blocked = defaultdict(int)

    def equity_now(ts):
        mtm = sum(p.notional * (1 + (last_price(p.symbol, ts, p.exit_price) / p.entry_price - 1)
                                 * (1 if p.side == "buy" else -1))
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
            if pos.family == "newbot":
                newbot_open -= 1
                stats = newbot_stats[pos.strategy_key]
                if pnl_dollars > 0:
                    stats["wins"] += 1
                    stats["win_pnls"].append(pnl_dollars)
                    stats["avg_win"] = float(np.mean(stats["win_pnls"]))
                else:
                    stats["losses"] += 1
                    stats["loss_pnls"].append(abs(pnl_dollars))
                    stats["avg_loss"] = float(np.mean(stats["loss_pnls"])) if stats["loss_pnls"] else 100.0
                newbot_pnl_hist[pos.strategy_key].append(pos.pnl_pct * 100)
                newbot_pnl_hist[pos.strategy_key] = newbot_pnl_hist[pos.strategy_key][-50:]
                dlg.record_pnl(pnl_dollars, equity_now(ts), now=ts)
            else:
                kargan2_open -= 1
            closed.append({"family": pos.family, "strategy_key": pos.strategy_key, "symbol": pos.symbol,
                            "pnl_dollars": pnl_dollars, "pnl_pct": pos.pnl_pct, "exit_time": ts})

        elif is_entry:
            equity = equity_now(ts)
            cb.check(equity)  # keep peak tracking current regardless of family

            if t["family"] == "kargan2":
                if kargan2_open >= KARGAN2_MAX_OPEN:
                    blocked["kargan2_max_open"] += 1; continue
                risk_amount = equity * 0.01
                stop_distance = t["stop_distance"]
                if stop_distance <= 0:
                    continue
                notional = min((risk_amount / stop_distance) * t["entry_price"], cash)
                if notional <= 0:
                    continue
                cash -= notional
                open_positions[tid] = OpenPos("kargan2", t["strategy_key"], t["symbol"], "buy",
                                               notional, t["entry_price"], t["exit_price"], t["pnl_pct"])
                kargan2_open += 1

            else:  # newbot
                if cb.is_triggered():
                    blocked["circuit_breaker"] += 1; continue
                if dlg.check(equity, now=ts):
                    blocked["daily_loss_gate"] += 1; continue
                if newbot_open >= NEWBOT_MAX_OPEN:
                    blocked["newbot_max_open"] += 1; continue

                s = newbot_stats[t["strategy_key"]]
                kelly = half_kelly_fraction(s["wins"], s["losses"], s["avg_win"], s["avg_loss"])
                if kelly <= 0:
                    kelly = NEWBOT_RISK_PCT
                approved, adj_size = mc_gate.pre_trade_check(newbot_pnl_hist[t["strategy_key"]], kelly)
                if not approved:
                    blocked["mc_gate"] += 1; continue

                notional = min(adj_size * equity, cash)
                if notional <= 0:
                    continue
                cash -= notional
                open_positions[tid] = OpenPos("newbot", t["strategy_key"], t["symbol"], t["side"],
                                               notional, t["entry_price"], t["exit_price"], t["pnl_pct"])
                newbot_open += 1

        equity_curve.append((ts, equity_now(ts)))

    return closed, pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date"), blocked, cb


def summarize(closed, eq_df, blocked, cb):
    print(f"\n{'='*70}")
    print("FULL SHARED-ACCOUNT SIMULATION -- Kargan2 (2 strategies) + New Trading bot (7 strategies)")
    print(f"{'='*70}")
    print(f"Starting equity: ${STARTING_EQUITY:,.2f} (same real 'Kargan 2' Alpaca account, both systems)")

    if eq_df.empty:
        print("No events processed.")
        return

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
    print(f"Circuit breaker tripped: {cb.is_triggered()}" +
          (f" (peak ${cb.peak_equity:,.2f})" if cb.is_triggered() else ""))

    if closed:
        n = len(closed)
        wins = sum(1 for c in closed if c["pnl_dollars"] > 0)
        total_pnl = sum(c["pnl_dollars"] for c in closed)
        print(f"\nClosed trades: {n} | Win rate: {wins/n:.1%} | Total realized PnL: ${total_pnl:+,.2f}")

        print(f"\n{'strategy':<24}{'trades':>8}{'win%':>8}{'PnL':>14}")
        print("-" * 54)
        by_key = defaultdict(list)
        for c in closed:
            by_key[(c["family"], c["strategy_key"])].append(c)
        for (family, key), trades in sorted(by_key.items(), key=lambda kv: -sum(c["pnl_dollars"] for c in kv[1])):
            wr = sum(1 for c in trades if c["pnl_dollars"] > 0) / len(trades)
            pnl = sum(c["pnl_dollars"] for c in trades)
            tag = "[Kargan2]" if family == "kargan2" else "[NewBOT]"
            print(f"{tag+' '+key:<24}{len(trades):>8}{wr*100:>7.1f}%{pnl:>+14,.2f}")

    if blocked:
        print(f"\nEntries blocked by risk gates: {dict(blocked)}")


if __name__ == "__main__":
    build_newbot_trades()
    build_kargan2_trades()
    build_price_series()
    closed, eq_df, blocked, cb = run_shared_ledger()
    summarize(closed, eq_df, blocked, cb)
