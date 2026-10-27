"""Cooperative deadlines and a process registry for bounded experiments."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterator
import uuid

import psutil


ROOT = Path(__file__).resolve().parents[2]


class DeadlineExceeded(RuntimeError):
    """The user-defined experiment deadline has been reached."""


class Deadline:
    def __init__(self, deadline: str | datetime | None = None):
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        if deadline is not None and deadline.tzinfo is None:
            raise ValueError("Deadline must include a timezone offset.")
        self.value = deadline

    def remaining_seconds(self) -> float:
        if self.value is None:
            return float("inf")
        return (self.value - datetime.now(timezone.utc)).total_seconds()

    def expired(self) -> bool:
        return self.remaining_seconds() <= 0

    def check(self) -> None:
        if self.expired():
            raise DeadlineExceeded(f"Deadline reached: {self.value.isoformat()}")


def atomic_replace(source: str | Path, destination: str | Path) -> None:
    """Replace a file atomically, retrying brief Windows permission locks.

    Eight attempts wait at most 1.27 seconds in total. Other filesystem errors
    propagate immediately, and persistent permission errors remain failures.
    """
    delays = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
    for attempt, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(delays) - 1:
                raise


def _atomic_json(path: Path, content: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2), encoding="utf-8")
    atomic_replace(temporary, path)


@contextmanager
def register_job(command: str | None = None, deadline: str | None = None,
                 metadata: dict | None = None) -> Iterator[Path]:
    """Register only this process; an external watchdog validates PID lifetime."""
    Deadline(deadline).check()
    registry = ROOT / ".runtime" / "jobs"
    registry.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    path = registry / f"{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
    record = {
        "pid": os.getpid(), "create_time": process.create_time(),
        "workspace": str(ROOT), "command": command or " ".join(sys.argv),
        "deadline": deadline, "status": "running",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    _atomic_json(path, record)
    try:
        yield path
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = type(exc).__name__
        raise
    else:
        record["status"] = "completed"
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(path, record)


def stop_registered_jobs(*, force: bool = False) -> list[int]:
    """Stop expired registered jobs, refusing recycled PIDs or other workspaces."""
    stopped = []
    for path in (ROOT / ".runtime" / "jobs").glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "running":
                continue
            if Path(record["workspace"]).resolve() != ROOT:
                continue
            if not force and not Deadline(record.get("deadline")).expired():
                continue
            process = psutil.Process(int(record["pid"]))
            if process.pid == os.getpid():
                continue
            if abs(process.create_time() - float(record["create_time"])) > 0.01:
                continue
            children = process.children(recursive=True)
            for child in children:
                child.terminate()
            process.terminate()
            _, alive = psutil.wait_procs(children + [process], timeout=3)
            for remaining in alive:
                remaining.kill()
            record["status"] = "stopped_by_deadline"
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(path, record)
            stopped.append(process.pid)
        except (OSError, ValueError, KeyError, psutil.Error):
            continue
    return stopped

