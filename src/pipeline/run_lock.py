from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .io import write_json


class RunLockError(RuntimeError):
    pass


FENCED_EXIT_CODE = 75


def run_lock_disabled() -> bool:
    raw = os.environ.get("PPIFLOW_NO_RUN_LOCK")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("r") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _owner_path(out_dir: Path) -> Path:
    return out_dir / ".ppiflow_lock" / "owner.json"


def _heartbeat_path(out_dir: Path) -> Path:
    return out_dir / ".ppiflow_lock" / "heartbeat"


def read_active_lock_id(out_dir: str | Path) -> Optional[str]:
    out_dir = Path(out_dir)
    owner = _read_json(_owner_path(out_dir))
    if not owner:
        return None
    lock_id = owner.get("lock_id")
    if not lock_id:
        return None
    return str(lock_id)


def ensure_expected_lock_id(
    out_dir: str | Path,
    *,
    wait_seconds: float = 5.0,
) -> Optional[str]:
    """
    Ensure this process has a stable expected lock id.

    If PPIFLOW_RUN_LOCK_ID is already set, we treat it as the expected fencing token.
    Otherwise, if OUT/.ppiflow_lock exists, we wait briefly for owner.json then set
    PPIFLOW_RUN_LOCK_ID to the current lock id.

    Returns the expected lock id (or None if no run lock is present/enabled).
    """
    if run_lock_disabled():
        return None
    existing = os.environ.get("PPIFLOW_RUN_LOCK_ID")
    if existing:
        return str(existing)

    out_dir = Path(out_dir)
    lock_dir = out_dir / ".ppiflow_lock"
    if not lock_dir.exists():
        return None

    deadline = time.time() + max(float(wait_seconds), 0.0)
    while True:
        active = read_active_lock_id(out_dir)
        if active:
            os.environ["PPIFLOW_RUN_LOCK_ID"] = str(active)
            return str(active)
        if time.time() >= deadline:
            raise RunLockError(f"Run lock present at {lock_dir} but owner.json is missing/unparseable")
        time.sleep(0.1)


def fence_process(message: str, *, exit_code: int = FENCED_EXIT_CODE) -> None:
    """
    Terminate this process because it no longer owns (or is consistent with) the active run lock.

    If called from a worker thread, we must hard-exit because raising SystemExit only kills
    that thread.
    """
    try:
        print(message, file=sys.__stdout__, flush=True)
    except Exception:
        pass
    if threading.current_thread() is threading.main_thread():
        raise SystemExit(exit_code)
    os._exit(exit_code)


def validate_expected_lock_id(out_dir: str | Path, *, expected_lock_id: str | None = None) -> None:
    if run_lock_disabled():
        return
    expected = expected_lock_id or os.environ.get("PPIFLOW_RUN_LOCK_ID")
    if not expected:
        return
    active = read_active_lock_id(out_dir)
    if not active:
        fence_process("[run_lock] active lock missing; exiting")
    if str(active) != str(expected):
        fence_process(
            f"[run_lock] lock changed (expected={expected} active={active}); exiting",
        )


@dataclass
class RunLock:
    out_dir: Path
    lock_id: str
    lock_dir: Path
    stale_after_seconds: float
    heartbeat_interval_seconds: float
    owner_payload: dict[str, Any]

    _stop: threading.Event
    _thread: threading.Thread | None = None

    @classmethod
    def acquire(
        cls,
        out_dir: str | Path,
        *,
        stale_after_seconds: float | None = None,
        heartbeat_interval_seconds: float = 5.0,
        steal: bool = False,
        disabled: bool = False,
        owner_extra: dict[str, Any] | None = None,
    ) -> Optional["RunLock"]:
        if disabled or run_lock_disabled():
            return None

        out_dir = Path(out_dir)
        lock_dir = out_dir / ".ppiflow_lock"
        owner_path = lock_dir / "owner.json"
        heartbeat_path = lock_dir / "heartbeat"

        interval = max(float(heartbeat_interval_seconds), 1.0)
        stale_after = float(stale_after_seconds) if stale_after_seconds is not None else max(30.0, 3.0 * interval)
        init_grace_seconds = 5.0

        while True:
            try:
                lock_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                # If owner.json is missing/unparseable, treat as "initializing" for a short grace window.
                owner = _read_json(owner_path)
                if owner is None:
                    try:
                        age = max(0.0, time.time() - lock_dir.stat().st_mtime)
                    except Exception:
                        age = init_grace_seconds + 1.0
                    if age <= init_grace_seconds:
                        time.sleep(0.2)
                        continue

                last_hb = None
                try:
                    last_hb = heartbeat_path.stat().st_mtime
                except Exception:
                    last_hb = None
                if last_hb is None:
                    try:
                        last_hb = lock_dir.stat().st_mtime
                    except Exception:
                        last_hb = None
                if last_hb is None:
                    last_hb = time.time()

                hb_age = max(0.0, time.time() - float(last_hb))
                is_stale = hb_age > stale_after
                if not is_stale and not steal:
                    host = owner.get("host") if isinstance(owner, dict) else None
                    pid = owner.get("pid") if isinstance(owner, dict) else None
                    updated_at = owner.get("updated_at") if isinstance(owner, dict) else None
                    raise RunLockError(
                        "Output directory is already locked by an active controller: "
                        f"{lock_dir} (host={host} pid={pid} updated_at={updated_at} hb_age_s={hb_age:.1f}). "
                        "If you're sure it's dead, wait for staleness or pass --steal-lock."
                    )

                old_lock_id = None
                if isinstance(owner, dict):
                    old_lock_id = owner.get("lock_id")
                ts = f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.{time.time_ns()}"
                archive = out_dir / f".ppiflow_lock.stale.{ts}.{old_lock_id or 'unknown'}"
                try:
                    lock_dir.rename(archive)
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    raise RunLockError(f"Failed to takeover run lock at {lock_dir}: {exc}") from exc
                continue

            # We own the directory now.
            lock_id = uuid.uuid4().hex
            now = _iso()
            payload: dict[str, Any] = {
                "lock_id": lock_id,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "started_at": now,
                "updated_at": now,
            }
            if owner_extra:
                # Keep keys shallow and JSON-serializable.
                payload.update({k: v for k, v in owner_extra.items() if k and v is not None})
            try:
                write_json(owner_path, payload, indent=2)
            except Exception as exc:
                raise RunLockError(f"Failed to write run lock owner.json at {owner_path}: {exc}") from exc
            try:
                heartbeat_path.write_text(str(time.time()))
            except Exception:
                pass

            os.environ["PPIFLOW_RUN_LOCK_ID"] = str(lock_id)

            lock = cls(
                out_dir=out_dir,
                lock_id=str(lock_id),
                lock_dir=lock_dir,
                stale_after_seconds=float(stale_after),
                heartbeat_interval_seconds=float(interval),
                owner_payload=dict(payload),
                _stop=threading.Event(),
            )
            lock._start_heartbeat()
            return lock

    def _start_heartbeat(self) -> None:
        owner_path = self.lock_dir / "owner.json"
        heartbeat_path = self.lock_dir / "heartbeat"

        def _loop() -> None:
            while not self._stop.wait(self.heartbeat_interval_seconds):
                # If the lock has been taken over, stop heartbeating (do not recreate directories).
                active = read_active_lock_id(self.out_dir)
                if not active or str(active) != str(self.lock_id):
                    fence_process(
                        f"[run_lock] lock changed (expected={self.lock_id} active={active}); exiting",
                    )
                try:
                    heartbeat_path.write_text(str(time.time()))
                except Exception:
                    pass
                try:
                    payload = dict(self.owner_payload)
                    payload["updated_at"] = _iso()
                    self.owner_payload = payload
                    write_json(owner_path, payload, indent=2)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        # Only remove the lock if we still appear to be the active owner.
        active = read_active_lock_id(self.out_dir)
        if active and str(active) != str(self.lock_id):
            return

        try:
            shutil.rmtree(self.lock_dir, ignore_errors=True)
        except Exception:
            pass
