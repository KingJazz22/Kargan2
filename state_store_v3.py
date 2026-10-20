"""Local JSON state for live_trading_v3.py -- its own file, separate from
live_state.json (v1) and live_state_v2.json (v2), which track other accounts.
"""
import json
import os

STATE_DIR = os.getenv("STATE_DIR_V3", os.path.dirname(__file__))
STATE_FILE = os.path.join(STATE_DIR, "live_state_v3.json")


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
