"""Local JSON state for the Breakout Hunter live runner.

Tracks per-symbol bookkeeping the broker doesn't know about: ATR at entry,
the initial structural stop, whether the trail has activated, the extreme
price since entry, and the current resting stop order's ID -- needed to
replicate the two-stage stop from the backtest (structural stop until 1x ATR
profit, then an ATR trail), which Alpaca has no native order type for.
"""
import json
import os

# STATE_DIR lets a deployment point this at a mounted persistent volume (e.g.
# Railway) instead of the code directory itself, which may be ephemeral.
STATE_DIR = os.getenv("STATE_DIR", os.path.dirname(__file__))
STATE_FILE = os.path.join(STATE_DIR, "live_state.json")


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
