"""Run the Adaptive Trend Following backtest on a US equity and a crypto symbol."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import run_backtest, summarize
from data import fetch_yfinance

START = "2019-01-01"
END = "2026-08-01"
STARTING_EQUITY = 100_000.0
RISK_PCT = 0.01

SYMBOLS = {
    "US market": "SPY",
    "Crypto": "BTC-USD",
}


def main():
    for label, symbol in SYMBOLS.items():
        print(f"\n=== {label}: {symbol} ({START} to {END}, daily) ===")
        df = fetch_yfinance(symbol, START, END)
        trades, equity_df = run_backtest(df, starting_equity=STARTING_EQUITY, risk_pct=RISK_PCT)
        stats = summarize(trades, equity_df, STARTING_EQUITY)

        if stats["num_trades"] == 0:
            print("No trades generated.")
            continue

        for k, v in stats.items():
            print(f"  {k}: {v}")

        fig, ax = plt.subplots(figsize=(10, 4))
        equity_df["equity"].plot(ax=ax, title=f"{symbol} — Adaptive Trend Following equity curve")
        ax.set_ylabel("Equity ($)")
        fig.tight_layout()
        out_path = f"equity_{symbol.replace('/', '_')}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  equity curve saved -> {out_path}")


if __name__ == "__main__":
    main()
