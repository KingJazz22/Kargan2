"""Persistent-process entry point for Railway. Runs live_trading.main() on
an internal schedule instead of relying on Railway's native cron-schedule
feature: that spins up a fresh, ephemeral container per trigger, which would
lose live_state.json (open positions, stop-order IDs, per-symbol cursors)
between runs unless a Volume is correctly wired to a cron-type service.
Keeping ONE long-lived process alive avoids that risk entirely, and mirrors
how New Trading bot's own Railway worker (workers/trading_worker.py, FastAPI
+ APScheduler) is already structured on this same Railway account.

Times below are fixed UTC and do NOT auto-adjust for US Daylight Saving Time
-- same caveat the local Windows Task Scheduler version had. Currently set
for EDT (UTC-4): 9:35/11:35/13:35/15:35 ET = 13:35/15:35/17:35/19:35 UTC.
Shift by +1 hour in UTC (14:35/16:35/18:35/20:35) for EST (~early November
to mid-March).
"""
import time
from datetime import datetime, timezone

import live_trading as lt

RUN_TIMES_UTC = ["13:35", "15:35", "17:35", "19:35"]
CHECK_INTERVAL_SECONDS = 30
RUN_WINDOW_SECONDS = 300  # tolerance so a check slightly after the mark still fires


def main_loop():
    last_run_key = None
    print(f"Kargan2 Railway runner started. Scheduled UTC times: {RUN_TIMES_UTC} (Mon-Fri)", flush=True)
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
                        print(f"live_trading.main() raised: {e}", flush=True)
                    last_run_key = key
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
