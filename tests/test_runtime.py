from datetime import datetime, timedelta, timezone
import json

import pytest

import coldstart.runtime as runtime


def test_deadline_requires_timezone_and_expires():
    with pytest.raises(ValueError, match="timezone"):
        runtime.Deadline("2030-01-01T12:00:00")
    deadline = runtime.Deadline(datetime.now(timezone.utc) - timedelta(seconds=1))
    with pytest.raises(runtime.DeadlineExceeded):
        deadline.check()
    assert runtime.Deadline(None).remaining_seconds() == float("inf")


def test_registry_records_success_and_failure_and_does_not_target_completed_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    with runtime.register_job("successful test") as path:
        record = json.loads(path.read_text())
        assert record["status"] == "running"
        assert record["create_time"] > 0
    assert json.loads(path.read_text())["status"] == "completed"
    with pytest.raises(RuntimeError, match="intentional"):
        with runtime.register_job("failing test") as failed:
            raise RuntimeError("intentional")
    assert json.loads(failed.read_text())["status"] == "failed"
    assert runtime.stop_registered_jobs(force=True) == []


def test_atomic_replace_retries_permission_locks_and_preserves_old_file_until_success(tmp_path, monkeypatch):
    source, destination = tmp_path / "new.tmp", tmp_path / "result.json"
    source.write_text("new")
    destination.write_text("old")
    original_replace = runtime.os.replace
    attempts, waits = [], []

    def briefly_locked(left, right):
        attempts.append(1)
        assert destination.read_text() == "old"
        if len(attempts) <= 2:
            raise PermissionError("simulated Windows reader lock")
        original_replace(left, right)

    monkeypatch.setattr(runtime.os, "replace", briefly_locked)
    monkeypatch.setattr(runtime.time, "sleep", waits.append)
    runtime.atomic_replace(source, destination)
    assert destination.read_text() == "new" and not source.exists()
    assert len(attempts) == 3 and waits == [0.01, 0.02]


def test_atomic_replace_remains_bounded_and_does_not_swallow_persistent_failure(tmp_path, monkeypatch):
    source, destination = tmp_path / "new.tmp", tmp_path / "result.json"
    source.write_text("new")
    destination.write_text("old")
    attempts, waits = [], []

    def locked(left, right):
        attempts.append(1)
        raise PermissionError("persistent permission failure")

    monkeypatch.setattr(runtime.os, "replace", locked)
    monkeypatch.setattr(runtime.time, "sleep", waits.append)
    with pytest.raises(PermissionError, match="persistent"):
        runtime.atomic_replace(source, destination)
    assert len(attempts) == 8
    assert sum(waits) == pytest.approx(1.27)
    assert destination.read_text() == "old" and source.read_text() == "new"
