"""Persistent-process entry point for Railway. Runs live_trading.main() on
an internal schedule instead of relying on Railway's native cron-schedule
feature: that spins up a fresh, ephemeral container per trigger, which would
lose live_state.json (open positions, stop-order IDs, per-symbol cursors)
between runs unless a Volume is correctly wired to a cron-type service.
Keeping ONE long-lived process alive avoids that risk entirely, and mirrors
how New Trading bot's own Railway worker (workers/trading_worker.py, FastAPI
+ APScheduler) is already structured on this same Railway account.

Also doubles as the entry point for the SECOND Railway service
(kargan2-live-trading-v2, the "master combined" bot on its own dedicated
paper account) via the RUNNER_TARGET env var -- set on that service only,
so it defaults to "v1" (this file's original, unchanged behavior) everywhere
else, including the already-running kargan2-live-trading service, which sets
no such variable.

Times below are fixed UTC and do NOT auto-adjust for US Daylight Saving Time
-- same caveat the local Windows Task Scheduler version had. Currently set
for EDT (UTC-4): 9:35/11:35/13:35/15:35 ET = 13:35/15:35/17:35/19:35 UTC.
Shift by +1 hour in UTC (14:35/16:35/18:35/20:35) for EST (~early November
to mid-March).
"""
import os
import time
from datetime import datetime, timezone

RUNNER_TARGET = os.getenv("RUNNER_TARGET", "v1")
if RUNNER_TARGET == "v2":
    import live_trading_v2 as lt
else:
    import live_trading as lt

RUN_TIMES_UTC = ["13:35", "15:35", "17:35", "19:35"]
CHECK_INTERVAL_SECONDS = 30
RUN_WINDOW_SECONDS = 300  # tolerance so a check slightly after the mark still fires


def main_loop():
    last_run_key = None
    print(f"Kargan2 Railway runner started (target={RUNNER_TARGET}). Scheduled UTC times: {RUN_TIMES_UTC} (Mon-Fri)", flush=True)
    while True:
        now = datetime.now(timezone.utc)
        if now.weekday() < 5:  # Monday=0 ... Friday=4
            for t in RUN_TIMES_UTC:
                hour, minute = int(t.split(":")[0]), int(t.split(":")[1])
                run_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                key = (now.date().isoformat(), t)
                elapsed = (now - run_dt).total_seconds()
                if 0 <= elapsed < RUN_WINDOW_SECONDS and key != last_run_key:
                    print(f"\n>>> Triggering scheduled run for {t} UTC (elapsed {elapsed:.0f}s)", flush=True)
                    try:
                        lt.main()
                    except Exception as e:
                        print(f"{lt.__name__}.main() raised: {e}", flush=True)
                    last_run_key = key
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
