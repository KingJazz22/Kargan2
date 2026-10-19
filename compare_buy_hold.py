"""swing_pcse_exit vs simply holding the market, with honest daily
mark-to-market (sweep_risk_levels.replay_shared_ledger(marks=...)).

Strategy: the deployed config (14xATR stop, 3xATR target), 2% risk, live
breaker rule (re-arm after 30 days, half risk), $9,995 start, on
  - live-40: the 40 symbols live_trading_v2 actually trades
  - S&P PIT: all 9,861 point-in-time S&P trades (FINDINGS_PCSE_FRESH_UNIVERSE.md)
Benchmarks over the same dates, total return (dividend-adjusted bars):
  SPY, RSP (equal-weight S&P 500 ETF), and SPY scaled to the strategy's own
  average invested fraction (the rest in cash at 0%, as the strategy's idle
  cash is in the ledger).
Writes research_metalabel/results_buy_hold.txt.
"""
import numpy as np
import pandas as pd

import backtest_metalabel_pcse as ml
import backtest_pcse_fresh_universe as fu
import sweep_risk_levels as sw
from basket import US_SYMBOLS
from test_mtf_oos_symbols import OOS_US_SYMBOLS

sw.CIRCUIT_BREAKER_COOLDOWN_DAYS, sw.CIRCUIT_BREAKER_REARM_RISK_MULT = 30, 0.5
LIVE40 = set(US_SYMBOLS + OOS_US_SYMBOLS)


def records(t):
    return [{"strategy_key": "swing_pcse_exit", "symbol": r.symbol,
             "entry_time": sw.to_utc(r.entry_time), "exit_time": sw.to_utc(r.exit_time),
             "entry_price": r.entry_price, "exit_price": r.exit_price,
             "pnl_pct": r.exit_price / r.entry_price - 1, "stop_distance": r.stop_distance} for r in t.itertuples()]


def daily_stats(eq, label):
    eq = eq.dropna()
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    return {"portfolio": label, "total_return%": (eq.iloc[-1] / eq.iloc[0] - 1) * 100,
            "CAGR%": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100,
            "vol%": r.std() * np.sqrt(252) * 100, "sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxDD%": dd * 100}


def main():
    log = open(ml.OUT_DIR / "results_buy_hold.txt", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    t_all = pd.read_pickle("_pcse_fresh_trades.pkl").sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    bars = fu.load_bars()
    close = {s: g.set_index("date")["Close"].sort_index() for s, g in bars.groupby("symbol")}
    marks = {s: close[s] for s in set(t_all["symbol"]) | {"SPY"} if s in close}

    for name, t in [("live-40", t_all[t_all["symbol"].isin(LIVE40)]), ("S&P PIT", t_all)]:
        recs = records(t)
        _, eq_old, *_ = sw.replay_shared_ledger(recs, 0.02)
        closed, eq, taken, bc, _ = sw.replay_shared_ledger(recs, 0.02, marks=marks)
        daily = eq[eq["invested"].notna()]
        strat = daily["equity"].copy()
        strat.index = strat.index.tz_localize(None).normalize()
        start, end = strat.index[0], strat.index[-1]
        exposure = (daily["invested"] / daily["equity"]).mean()

        rows = [daily_stats(strat, f"swing_pcse ({name}), honest daily marks")]
        old = eq_old["equity"].resample("1D").last().dropna()
        old.index = old.index.tz_localize(None).normalize()
        rows.append(daily_stats(old[(old.index >= start) & (old.index <= end)], "  same, OLD look-ahead ledger"))
        for b in ["SPY", "RSP"]:
            px = close[b].loc[start:end]
            rows.append(daily_stats(px / px.iloc[0] * sw.STARTING_EQUITY, f"{b} buy & hold"))
        spy = close["SPY"].loc[start:end].pct_change().fillna(0)
        rows.append(daily_stats(sw.STARTING_EQUITY * (1 + exposure * spy).cumprod(),
                                f"{exposure:.0%} SPY + {1 - exposure:.0%} cash (strategy's avg exposure)"))
        say(f"\n=== {name}: {start.date()} -> {end.date()}, {taken} trades taken, "
            f"{bc} entries blocked by breaker, avg invested {exposure:.0%} ===")
        say(pd.DataFrame(rows).round(2).to_string(index=False))

        sr = strat.pct_change().dropna()
        spr = close["SPY"].reindex(sr.index).pct_change().dropna()
        j = sr.index.intersection(spr.index)
        beta = np.cov(sr[j], spr[j])[0, 1] / spr[j].var()
        say(f"daily correlation with SPY {sr[j].corr(spr[j]):.2f}, beta {beta:.2f}")
        yr = pd.DataFrame({"strategy": strat.resample("YE").last().pct_change() * 100,
                           "SPY": close["SPY"].loc[start:end].resample("YE").last().pct_change() * 100})
        yr.iloc[0] = [(strat.resample("YE").last().iloc[0] / strat.iloc[0] - 1) * 100,
                      (close["SPY"].loc[start:end].resample("YE").last().iloc[0] / close["SPY"].loc[start] - 1) * 100]
        yr.index = yr.index.year
        say("calendar-year returns %:\n" + yr.round(1).T.to_string())
    log.close()


if __name__ == "__main__":
    main()
