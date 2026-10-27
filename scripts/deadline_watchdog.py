"""Stop only registered project jobs at a user-specified hard deadline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coldstart.runtime import Deadline, ROOT, stop_registered_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--stop-now", action="store_true")
    args = parser.parse_args()
    deadline = Deadline(args.deadline)
    log = ROOT / ".runtime" / "watchdog.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", buffering=1) as stream:
        while True:
            final = args.stop_now or deadline.expired()
            stopped = stop_registered_jobs(force=final)
            if stopped or final:
                stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                         "stopped_pids": stopped, "final": final}) + "\n")
            if final or (ROOT / ".runtime" / "STOP_WATCHDOG").exists():
                return
            time.sleep(min(2, max(0.01, deadline.remaining_seconds())))


if __name__ == "__main__":
    main()

