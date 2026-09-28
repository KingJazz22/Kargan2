"""Local JSON state for live_trading_v2.py -- deliberately a separate file
from state_store.py's live_state.json, since that file already tracks open
positions/stop-order IDs for a DIFFERENT Alpaca account (broker_alpaca.py's).
Mixing the two would corrupt both runners' reconciliation logic.
"""
import json
import os

STATE_DIR = os.getenv("STATE_DIR_V2", os.path.dirname(__file__))
STATE_FILE = os.path.join(STATE_DIR, "live_state_v2.json")


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
