"""Each live bot's whole book vs buy-and-hold, with honest daily
mark-to-market (sweep_risk_levels.replay_shared_ledger(marks=...)).

  v1 book: Breakout Hunter + Swing-Structure Breakout (40 live symbols) + PCSE (20)
  v2 book: swing_pcse_exit (40 live symbols) + PCSE (20)   [Breakout Hunter entries disabled]
  plus each strategy alone.

Trades: real-entry trades from random_entry_all_bots.py (_real_trades_all_bots.pkl)
and _pcse_fresh_trades.pkl for swing_pcse_exit. One shared account, 2% risk,
live breaker rule, $9,995 start. Not modeled: v1's partial profit-taking on
Breakout Hunter, Orland EWZ (no backtest engine), the v3 paper bot.
Writes research_metalabel/results_bots_buy_hold.txt.
"""
import pickle

import numpy as np
import pandas as pd

import backtest_metalabel_pcse as ml
import backtest_pcse_fresh_universe as fu
import compare_buy_hold as cb
import sweep_risk_levels as sw

sw.CIRCUIT_BREAKER_COOLDOWN_DAYS, sw.CIRCUIT_BREAKER_REARM_RISK_MULT = 30, 0.5


def recs(df, key):
    return [{"strategy_key": key, "symbol": r.symbol, "entry_time": sw.to_utc(r.entry_time),
             "exit_time": sw.to_utc(r.exit_time), "entry_price": r.entry_price, "exit_price": r.exit_price,
             "pnl_pct": r.exit_price / r.entry_price - 1, "stop_distance": r.stop_distance}
            for r in df.itertuples() if r.stop_distance and r.stop_distance > 0]


def main():
    log = open(ml.OUT_DIR / "results_bots_buy_hold.txt", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    real = pickle.load(open("_real_trades_all_bots.pkl", "rb"))
    live40 = lambda d: d[d["symbol"].isin(cb.LIVE40)]
    bo, ss, pc = live40(real["breakout_hunter"]), live40(real["swing_structure"]), real["pcse"]
    sp = pd.read_pickle("_pcse_fresh_trades.pkl")
    sp = sp[sp["symbol"].isin(cb.LIVE40)]
    books = {
        "Breakout Hunter alone": recs(bo, "breakout"),
        "Swing-Structure alone": recs(ss, "swing"),
        "PCSE alone": recs(pc, "pcse"),
        "swing_pcse_exit alone": recs(sp, "swing_pcse"),
        "v1 book (BH + Swing-Structure + PCSE)": recs(bo, "breakout") + recs(ss, "swing") + recs(pc, "pcse"),
        "v2 book (swing_pcse_exit + PCSE)": recs(sp, "swing_pcse") + recs(pc, "pcse"),
    }
    bars = fu.load_bars()
    close = {s: g.set_index("date")["Close"].sort_index() for s, g in bars.groupby("symbol")}
    syms = {r["symbol"] for b in books.values() for r in b} | {"SPY"}
    marks = {s: close[s] for s in syms if s in close}

    rows = []
    for name, rs in books.items():
        rs = sorted(rs, key=lambda r: r["entry_time"])
        _, eq, taken, bc, _ = sw.replay_shared_ledger(rs, 0.02, marks=marks)
        daily = eq[eq["invested"].notna()]
        e = daily["equity"].copy()
        e.index = e.index.tz_localize(None).normalize()
        e = e[~e.index.duplicated(keep="last")]
        exposure = (daily["invested"] / daily["equity"]).mean()
        st = cb.daily_stats(e, name)
        spy = close["SPY"].loc[e.index[0]:e.index[-1]]
        b = cb.daily_stats(spy / spy.iloc[0], "SPY")
        m = cb.daily_stats((1 + exposure * spy.pct_change().fillna(0)).cumprod(), "matched")
        sr, spr = e.pct_change().dropna(), spy.pct_change().reindex(e.index).dropna()
        j = sr.index.intersection(spr.index)
        rows.append({"book": name, "from": e.index[0].date(), "taken": taken, "blocked": bc,
                     "invested%": exposure * 100, "CAGR%": st["CAGR%"], "sharpe": st["sharpe"],
                     "maxDD%": st["maxDD%"], "SPY_CAGR%": b["CAGR%"], "SPY_maxDD%": b["maxDD%"],
                     "matchedSPY_CAGR%": m["CAGR%"], "matchedSPY_sharpe": m["sharpe"],
                     "beta": np.cov(sr[j], spr[j])[0, 1] / spr[j].var(), "corr": sr[j].corr(spr[j])})
    say(pd.DataFrame(rows).round(2).to_string(index=False))
    log.close()


if __name__ == "__main__":
    main()
