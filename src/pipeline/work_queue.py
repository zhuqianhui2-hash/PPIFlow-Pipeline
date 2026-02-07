from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .io import ensure_dir
from .run_lock import RunLockError, ensure_expected_lock_id, fence_process, read_active_lock_id, run_lock_disabled


class WorkQueueError(RuntimeError):
    pass


@dataclass
class WorkItem:
    id: str
    payload: dict
    outputs: list[str] | None = None


@dataclass
class ClaimedItem:
    item: WorkItem
    attempt: int


@dataclass
class WaitResult:
    status: str
    reason: str | None = None
    counts: Dict[str, int] | None = None
    progress: Dict[str, Any] | None = None
    failed: bool = False
    complete: bool = False


def _iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _safe_id(raw: str) -> str:
    raw = str(raw)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    if not safe:
        safe = hashlib.sha256(raw.encode()).hexdigest()[:16]
    if len(safe) > 128:
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        safe = safe[:64] + "_" + h
    return safe


def _warn(message: str) -> None:
    print(f"[work_queue] WARN {message}", file=sys.__stdout__, flush=True)


class WorkQueue:
    def __init__(
        self,
        out_dir: str | Path,
        step: str,
        cfg: Dict[str, Any] | None = None,
        *,
        worker_id: Optional[str] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.step = str(step)
        self.cfg = cfg or {}
        self.base_dir = self.out_dir / ".work" / self.step
        self.db_path = self.base_dir / "queue.db"

        self.lease_seconds = int(self.cfg.get("lease_seconds") or 300)
        self.max_attempts = int(self.cfg.get("max_attempts") or 2)
        self.retry_failed = bool(self.cfg.get("retry_failed"))
        batch_val = self.cfg.get("batch_size")
        self.batch_size = 1 if batch_val is None else int(batch_val)
        self.leader_timeout = int(self.cfg.get("leader_timeout") or 600)
        allow_reuse = self.cfg.get("allow_reuse")
        if allow_reuse is None:
            allow_reuse = True
        self.allow_reuse = bool(allow_reuse)
        self.wait_timeout = self.cfg.get("wait_timeout")
        if self.wait_timeout is not None:
            try:
                self.wait_timeout = float(self.wait_timeout)
            except Exception:
                self.wait_timeout = None
        self.sqlite_journal_mode = str(self.cfg.get("sqlite_journal_mode") or "WAL").upper()
        self.busy_timeout_ms = int(self.cfg.get("busy_timeout_ms") or 5000)

        self.worker_id = worker_id or self._default_worker_id()
        # Snapshot expected lock id if provided; otherwise resolve lazily on first DB mutation.
        self._expected_run_lock_id: str | None = os.environ.get("PPIFLOW_RUN_LOCK_ID")

    def _validate_run_lock(self) -> None:
        """
        Fencing token validation.

        Any DB write that affects correctness must ensure this worker is still operating under
        the active run lock. If the lock id changes (takeover), we should stop mutating state.
        """
        if run_lock_disabled():
            return
        expected = self._expected_run_lock_id
        if not expected:
            try:
                expected = ensure_expected_lock_id(self.out_dir, wait_seconds=5.0)
            except RunLockError as exc:
                raise WorkQueueError(str(exc)) from exc
            self._expected_run_lock_id = expected
        if not expected:
            return
        active = read_active_lock_id(self.out_dir)
        if not active:
            fence_process("[work_queue] run lock missing; exiting")
        if str(active) != str(expected):
            fence_process(
                f"[work_queue] run lock changed (expected={expected} active={active}); exiting",
            )

    @classmethod
    def from_step_dir(cls, step_dir: Path) -> "WorkQueue":
        step_dir = Path(step_dir)
        out_dir = step_dir.parent.parent
        step = step_dir.name
        return cls(out_dir, step, cfg={})

    def _default_worker_id(self) -> str:
        rank = os.environ.get("RANK", "0")
        return f"{socket.gethostname()}:{os.getpid()}:{rank}"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA journal_mode = {self.sqlite_journal_mode}")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        return conn

    def _drop_db_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _rebuild_lock_path(self) -> Path:
        return self.base_dir / "rebuild.lock"

    def _rebuild_lock_max_age(self) -> float:
        raw = self.cfg.get("rebuild_lock_max_age")
        if raw is None:
            raw = self.leader_timeout or 600
        try:
            return float(raw)
        except Exception:
            return float(self.leader_timeout or 600)

    def _rebuild_lock_age(self) -> Optional[float]:
        path = self._rebuild_lock_path()
        now = time.time()
        try:
            data = path.read_text().strip()
            if data:
                ts_raw = data.split("|", 1)[0]
                ts = float(ts_raw)
                return max(0.0, now - ts)
        except Exception:
            pass
        try:
            return max(0.0, now - path.stat().st_mtime)
        except Exception:
            return None

    def _acquire_rebuild_lock(self) -> bool:
        ensure_dir(self.base_dir)
        path = self._rebuild_lock_path()
        now = time.time()
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            age = self._rebuild_lock_age()
            max_age = self._rebuild_lock_max_age()
            if age is not None and age > max_age:
                tmp = path.with_suffix(".tmp")
                try:
                    tmp.write_text(f"{now}|{self.worker_id}")
                    os.replace(tmp, path)
                    return True
                except Exception:
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
            return False
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{now}|{self.worker_id}")
        return True

    def _release_rebuild_lock(self) -> None:
        try:
            self._rebuild_lock_path().unlink()
        except Exception:
            pass

    def _wait_for_rebuild(self, timeout: float = 600.0) -> bool:
        start = time.time()
        while True:
            if not self._rebuild_lock_path().exists() and self.db_path.exists():
                return False
            age = self._rebuild_lock_age()
            if age is not None and age > self._rebuild_lock_max_age():
                if self._acquire_rebuild_lock():
                    return True
            if (time.time() - start) >= timeout:
                raise WorkQueueError(f"Timed out waiting for rebuild of {self.step}")
            time.sleep(0.5)

    def _ensure_db(self) -> None:
        ensure_dir(self.base_dir)
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT,
                    status TEXT,
                    attempts INTEGER,
                    last_error TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    heartbeat_ts REAL,
                    lease_seconds INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    item_id TEXT,
                    attempt INTEGER,
                    status TEXT,
                    error TEXT,
                    ended_at TEXT,
                    worker_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS leader (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    worker_id TEXT,
                    heartbeat_ts REAL,
                    lease_seconds INTEGER,
                    status TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_items_status ON items(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_items_updated ON items(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_heartbeat ON claims(heartbeat_ts)")
        finally:
            conn.close()

    def _read_meta(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            return {}
        conn = self._connect()
        try:
            rows = conn.execute("SELECT key, value_json FROM meta").fetchall()
            out: Dict[str, Any] = {}
            for row in rows:
                key = row["key"]
                val = row["value_json"]
                try:
                    out[key] = json.loads(val) if val is not None else None
                except Exception:
                    out[key] = val
            return out
        finally:
            conn.close()

    def _write_meta(self, meta: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            for key, value in meta.items():
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value_json) VALUES (?, ?)",
                    (str(key), json.dumps(value, separators=(",", ":"))),
                )
        finally:
            conn.close()

    def _validate_or_write_meta(self, meta: Dict[str, Any]) -> None:
        existing = self._read_meta()
        if not existing:
            self._write_meta(meta)
            return
        mismatched = {}
        for key, value in meta.items():
            if key not in existing:
                continue
            if existing.get(key) != value:
                mismatched[key] = {"existing": existing.get(key), "incoming": value}
        if mismatched:
            if self.allow_reuse:
                _warn(f"meta mismatch for {self.step}: {list(mismatched.keys())}")
            else:
                raise WorkQueueError(
                    f"Work queue meta mismatch for {self.step}. Use --work-queue-reuse to override."
                )
        # fill missing keys if possible
        missing = {k: v for k, v in meta.items() if k not in existing}
        if missing:
            self._write_meta({**existing, **missing})

    def _items_fingerprint(self, items: Iterable[WorkItem]) -> str:
        payload = []
        for item in items:
            payload.append(
                {
                    "id": _safe_id(item.id),
                    "payload": item.payload,
                    "outputs": item.outputs or [],
                }
            )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _normalize_items(self, items: Iterable[WorkItem]) -> list[WorkItem]:
        normalized: list[WorkItem] = []
        for item in items:
            safe = _safe_id(item.id)
            if safe != item.id:
                payload = dict(item.payload)
                payload.setdefault("_orig_id", item.id)
                item = WorkItem(id=safe, payload=payload, outputs=item.outputs)
            normalized.append(item)
        return normalized

    def _work_queue_options(self) -> Dict[str, Any]:
        return {
            "lease_seconds": int(self.lease_seconds),
            "max_attempts": int(self.max_attempts),
            "retry_failed": bool(self.retry_failed),
            "batch_size": int(self.batch_size),
            "leader_timeout": int(self.leader_timeout),
            "wait_timeout": self.wait_timeout,
            "allow_reuse": bool(self.allow_reuse),
            "sqlite_journal_mode": self.sqlite_journal_mode,
        }

    def init_leader(self, meta: Dict[str, Any]) -> None:
        self._validate_run_lock()
        self._ensure_db()
        meta = dict(meta)
        meta.setdefault("schema_version", 1)
        meta.setdefault("work_queue_options", self._work_queue_options())
        self._validate_or_write_meta(meta)

    def init_items(
        self,
        items: Iterable[WorkItem],
        meta: Dict[str, Any],
        *,
        rebuild: bool = False,
        item_done_fn: Optional[callable] = None,
    ) -> None:
        self._validate_run_lock()
        lock_acquired = False
        if rebuild:
            if self._acquire_rebuild_lock():
                lock_acquired = True
                self._drop_db_files()
            else:
                timeout = float(self.wait_timeout or self.leader_timeout or 600)
                stole = self._wait_for_rebuild(timeout=timeout)
                if stole:
                    lock_acquired = True
                    self._drop_db_files()
                else:
                    rebuild = False
        try:
            self._ensure_db()
            incoming = list(items)
            normalized = self._normalize_items(incoming)
            meta = dict(meta)
            meta.setdefault("schema_version", 1)
            meta.setdefault("work_queue_options", self._work_queue_options())
            if normalized:
                meta["items_hash"] = self._items_fingerprint(normalized)
            self._validate_or_write_meta(meta)

            conn = self._connect()
            try:
                # Serialize initial item insertion across ranks/workers. Without this, two workers can
                # both observe COUNT(*) == 0 and concurrently INSERT, triggering UNIQUE violations.
                conn.execute("BEGIN IMMEDIATE")
                try:
                    count = conn.execute("SELECT COUNT(*) AS c FROM items").fetchone()["c"]
                    if count == 0:
                        if not normalized:
                            raise WorkQueueError(f"No work items provided for {self.step}")
                        for item in normalized:
                            payload = {
                                "id": item.id,
                                "payload": item.payload,
                                "outputs": item.outputs or [],
                            }
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO items(id, payload_json, status, attempts, last_error, updated_at)
                                VALUES (?, ?, 'pending', 0, NULL, ?)
                                """,
                                (item.id, json.dumps(payload, separators=(",", ":")), _iso()),
                            )
                    else:
                        if normalized:
                            incoming_hash = self._items_fingerprint(normalized)
                            existing_meta = self._read_meta()
                            existing_hash = existing_meta.get("items_hash")
                            if existing_hash and incoming_hash != existing_hash:
                                if self.allow_reuse:
                                    _warn(
                                        f"items hash mismatch for {self.step}: existing={existing_hash} incoming={incoming_hash}"
                                    )
                                else:
                                    raise WorkQueueError(
                                        "Work queue items hash mismatch for "
                                        f"{self.step}: existing={existing_hash} incoming={incoming_hash}"
                                    )
                    conn.execute("COMMIT")
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
            finally:
                conn.close()

            if rebuild and item_done_fn is not None:
                # Mark done items based on existing outputs.
                conn = self._connect()
                try:
                    rows = conn.execute("SELECT id, payload_json FROM items").fetchall()
                    for row in rows:
                        item_id = row["id"]
                        payload_json = row["payload_json"] or "{}"
                        try:
                            payload = json.loads(payload_json)
                        except Exception:
                            payload = {}
                        work_item = WorkItem(id=item_id, payload=dict(payload.get("payload") or {}))
                        try:
                            if item_done_fn(work_item):
                                conn.execute(
                                    "UPDATE items SET status='done', attempts=0, last_error=NULL, updated_at=? WHERE id=?",
                                    (_iso(), item_id),
                                )
                        except Exception:
                            continue
                finally:
                    conn.close()
        finally:
            if lock_acquired:
                self._release_rebuild_lock()

    def _claim_stale(self, heartbeat_ts: Optional[float], lease_seconds: Optional[int]) -> bool:
        lease = int(lease_seconds or self.lease_seconds)
        if heartbeat_ts is None:
            return True
        return (time.time() - float(heartbeat_ts)) > lease

    def claim_next(self) -> Optional[ClaimedItem]:
        if not self.db_path.exists():
            return None
        self._validate_run_lock()
        conn = self._connect()
        now = time.time()
        try:
            conn.execute("BEGIN IMMEDIATE")
            stale_rows = conn.execute(
                """
                SELECT items.id, items.attempts
                FROM items
                LEFT JOIN claims ON claims.id = items.id
                WHERE items.status = 'running'
                  AND items.attempts >= ?
                  AND (claims.heartbeat_ts IS NULL OR (? - claims.heartbeat_ts) > COALESCE(claims.lease_seconds, ?))
                """,
                (int(self.max_attempts), now, int(self.lease_seconds)),
            ).fetchall()
            if stale_rows:
                for row in stale_rows:
                    item_id = row["id"]
                    attempts = int(row["attempts"] or 0)
                    conn.execute(
                        "UPDATE items SET status='failed', last_error=?, updated_at=? WHERE id=?",
                        ("max_attempts exceeded", _iso(), item_id),
                    )
                    conn.execute(
                        "INSERT INTO attempts(item_id, attempt, status, error, ended_at, worker_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (item_id, attempts, "failed", "max_attempts exceeded", _iso(), self.worker_id),
                    )
                    conn.execute("DELETE FROM claims WHERE id=?", (item_id,))
            row = conn.execute(
                """
                SELECT items.id, items.payload_json, items.status, items.attempts,
                       claims.heartbeat_ts, claims.lease_seconds
                FROM items
                LEFT JOIN claims ON claims.id = items.id
                WHERE (
                    (items.status = 'pending' AND items.attempts < ?)
                    OR (
                        items.status IN ('failed', 'blocked')
                        AND ? = 1
                        AND items.attempts < ?
                    )
                    OR (
                        items.status = 'running'
                        AND (claims.heartbeat_ts IS NULL OR (? - claims.heartbeat_ts) > COALESCE(claims.lease_seconds, ?))
                        AND items.attempts < ?
                    )
                )
                ORDER BY items.updated_at ASC, items.id ASC
                LIMIT 1
                """,
                (
                    int(self.max_attempts),
                    1 if self.retry_failed else 0,
                    int(self.max_attempts),
                    now,
                    int(self.lease_seconds),
                    int(self.max_attempts),
                ),
            ).fetchone()
            if not row:
                conn.execute("COMMIT")
                return None
            item_id = row["id"]
            attempts = int(row["attempts"] or 0) + 1
            conn.execute(
                "UPDATE items SET status='running', attempts=?, last_error=NULL, updated_at=? WHERE id=?",
                (attempts, _iso(), item_id),
            )
            conn.execute(
                "INSERT OR REPLACE INTO claims(id, worker_id, heartbeat_ts, lease_seconds) VALUES (?, ?, ?, ?)",
                (item_id, self.worker_id, now, int(self.lease_seconds)),
            )
            conn.execute("COMMIT")
            payload_json = row["payload_json"] or "{}"
            try:
                payload = json.loads(payload_json)
            except Exception:
                payload = {}
            work_item = WorkItem(
                id=item_id,
                payload=dict(payload.get("payload") or {}),
                outputs=list(payload.get("outputs") or []) or None,
            )
            return ClaimedItem(item=work_item, attempt=attempts)
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def heartbeat(self, item_id: str) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE claims SET heartbeat_ts=? WHERE id=?",
                (time.time(), str(item_id)),
            )
        finally:
            conn.close()

    def touch_items(self, item_ids: Iterable[str]) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        ids = [str(i) for i in item_ids if i]
        if not ids:
            return
        conn = self._connect()
        now = time.time()
        try:
            for idx in range(0, len(ids), 200):
                chunk = ids[idx : idx + 200]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE claims SET heartbeat_ts=? WHERE worker_id=? AND id IN ({placeholders})",
                    (now, self.worker_id, *chunk),
                )
        finally:
            conn.close()

    def mark_done(self, item_id: str, attempt: int, *, note: Optional[str] = None) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE items SET status='done', attempts=?, last_error=NULL, updated_at=? WHERE id=?",
                (int(attempt), _iso(), str(item_id)),
            )
            conn.execute(
                "INSERT INTO attempts(item_id, attempt, status, error, ended_at, worker_id) VALUES (?, ?, ?, ?, ?, ?)",
                (str(item_id), int(attempt), "done", note, _iso(), self.worker_id),
            )
            conn.execute("DELETE FROM claims WHERE id=?", (str(item_id),))
        finally:
            conn.close()

    def mark_failed(self, item_id: str, attempt: int, error: str) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE items SET status='failed', attempts=?, last_error=?, updated_at=? WHERE id=?",
                (int(attempt), str(error), _iso(), str(item_id)),
            )
            conn.execute(
                "INSERT INTO attempts(item_id, attempt, status, error, ended_at, worker_id) VALUES (?, ?, ?, ?, ?, ?)",
                (str(item_id), int(attempt), "failed", str(error), _iso(), self.worker_id),
            )
            conn.execute("DELETE FROM claims WHERE id=?", (str(item_id),))
        finally:
            conn.close()

    def mark_failed_items(self, item_ids: Iterable[str], *, reason: str = "prior failure") -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        ids = [str(i) for i in item_ids if i]
        if not ids:
            return
        conn = self._connect()
        now = _iso()
        try:
            for idx in range(0, len(ids), 200):
                chunk = ids[idx : idx + 200]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE items SET status='failed', last_error=?, updated_at=? WHERE id IN ({placeholders})",
                    (str(reason), now, *chunk),
                )
        finally:
            conn.close()

    def mark_blocked(self, item_id: str, attempt: int, reason: str) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE items SET status='blocked', attempts=?, last_error=?, updated_at=? WHERE id=?",
                (int(attempt), str(reason), _iso(), str(item_id)),
            )
            conn.execute(
                "INSERT INTO attempts(item_id, attempt, status, error, ended_at, worker_id) VALUES (?, ?, ?, ?, ?, ?)",
                (str(item_id), int(attempt), "blocked", str(reason), _iso(), self.worker_id),
            )
            conn.execute("DELETE FROM claims WHERE id=?", (str(item_id),))
        finally:
            conn.close()

    def release_worker_claims(self) -> int:
        """
        Best-effort cleanup for graceful shutdowns.

        When a worker process exits (SIGTERM/SIGINT, preemption, timeout), any in-flight items it
        previously claimed can otherwise remain stuck as status='running' until the lease expires.
        Releasing those claims allows immediate resume without waiting for lease_seconds.

        Returns number of claims released.
        """
        if not self.db_path.exists():
            return 0
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id FROM claims WHERE worker_id=?",
                (self.worker_id,),
            ).fetchall()
            item_ids = [str(r["id"]) for r in rows]
            if not item_ids:
                conn.execute("COMMIT")
                return 0
            # Do not flip running -> pending here: pending items are only claimable when
            # attempts < max_attempts, so this can wedge resume if an interrupted claim
            # consumed the final attempt. Deleting claims is sufficient to make running
            # items immediately reclaimable (NULL claim heartbeat is treated as stale).
            conn.execute("DELETE FROM claims WHERE worker_id=?", (self.worker_id,))
            conn.execute("COMMIT")
            return len(item_ids)
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            return 0
        finally:
            conn.close()

    def counts(self) -> Dict[str, int]:
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "blocked": 0}
        if not self.db_path.exists():
            return counts
        conn = self._connect()
        try:
            rows = conn.execute("SELECT status, COUNT(*) AS c FROM items GROUP BY status").fetchall()
            for row in rows:
                status = str(row["status"] or "pending")
                if status not in counts:
                    status = "pending"
                counts[status] = int(row["c"] or 0)
        finally:
            conn.close()
        return counts

    def iter_items(self, status: Optional[str] = None) -> list[tuple[WorkItem, str]]:
        if not self.db_path.exists():
            return []
        conn = self._connect()
        try:
            if status:
                rows = conn.execute(
                    "SELECT id, payload_json, status FROM items WHERE status=?",
                    (str(status),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT id, payload_json, status FROM items").fetchall()
            items: list[tuple[WorkItem, str]] = []
            for row in rows:
                payload_json = row["payload_json"] or "{}"
                try:
                    payload = json.loads(payload_json)
                except Exception:
                    payload = {}
                work_item = WorkItem(
                    id=row["id"],
                    payload=dict(payload.get("payload") or {}),
                    outputs=list(payload.get("outputs") or []) or None,
                )
                items.append((work_item, str(row["status"] or "pending")))
            return items
        finally:
            conn.close()

    def progress(self) -> Dict[str, Any]:
        counts = self.counts()
        expected_total = sum(counts.values())
        produced_total = counts.get("done", 0) + counts.get("failed", 0) + counts.get("blocked", 0)
        if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
            status = "completed"
        else:
            status = "running"
        return {
            "expected_total": expected_total,
            "produced_total": produced_total,
            "status": status,
            "counts": counts,
            "updated_at": _iso(),
        }

    def leader_status(self) -> Optional[Dict[str, Any]]:
        if not self.db_path.exists():
            return None
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM leader WHERE id=1").fetchone()
            if not row:
                return None
            return {
                "worker_id": row["worker_id"],
                "heartbeat_ts": row["heartbeat_ts"],
                "lease_seconds": row["lease_seconds"],
                "status": row["status"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def acquire_leader(self) -> bool:
        self._validate_run_lock()
        self._ensure_db()
        conn = self._connect()
        now = time.time()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM leader WHERE id=1").fetchone()
            if row:
                status = str(row["status"] or "")
                if status == "completed":
                    conn.execute("COMMIT")
                    return False
                if status == "failed" and not self.retry_failed:
                    conn.execute("COMMIT")
                    return False
                heartbeat_ts = row["heartbeat_ts"]
                lease_seconds = row["lease_seconds"]
                if not self._claim_stale(heartbeat_ts, lease_seconds):
                    conn.execute("COMMIT")
                    return False
            conn.execute(
                "INSERT OR REPLACE INTO leader(id, worker_id, heartbeat_ts, lease_seconds, status, updated_at) VALUES (1, ?, ?, ?, ?, ?)",
                (self.worker_id, now, int(self.leader_timeout), "running", _iso(now)),
            )
            conn.execute("COMMIT")
            return True
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def leader_heartbeat(self) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE leader SET heartbeat_ts=?, updated_at=? WHERE id=1 AND worker_id=?",
                (time.time(), _iso(), self.worker_id),
            )
        finally:
            conn.close()

    def write_complete(self) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE leader SET status='completed', updated_at=? WHERE id=1",
                (_iso(),),
            )
        finally:
            conn.close()

    def write_failed(self, *, error: str | None = None) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE leader SET status='failed', updated_at=? WHERE id=1",
                (_iso(),),
            )
        finally:
            conn.close()

    def release_leader(self) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            row = conn.execute("SELECT status FROM leader WHERE id=1").fetchone()
            status = str(row["status"] or "") if row else ""
            if status in {"completed", "failed"}:
                return
            conn.execute("DELETE FROM leader WHERE id=1 AND worker_id=?", (self.worker_id,))
        finally:
            conn.close()

    def reset_items_for_retry(self) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE items SET status='pending', attempts=0, last_error=NULL, updated_at=? WHERE status IN ('failed','blocked','running')",
                (_iso(),),
            )
            conn.execute("DELETE FROM claims")
        finally:
            conn.close()

    def reset_leader_for_retry(self) -> None:
        if not self.db_path.exists():
            return
        self._validate_run_lock()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM leader WHERE id=1")
        finally:
            conn.close()

    def attempt_log_path(self, item_id: str, attempt: int) -> Path:
        # Per-worker log file (prefixed lines handled by caller).
        fname = f"worker-{socket.gethostname()}_{os.getpid()}_{os.environ.get('RANK', '0')}.log"
        return self.base_dir / fname


def reset_all_claims_and_leaders(out_dir: str | Path) -> Dict[str, Any]:
    """
    Controller-only recovery: clear claims + leader rows across all step queue.db files.

    This enables immediate resume after hard-kill without waiting for lease_seconds/leader_timeout
    to expire. It intentionally does NOT change items.status or attempts.

    Returns a small summary for logging/debugging.
    """
    out_dir = Path(out_dir)
    work_root = out_dir / ".work"
    summary: Dict[str, Any] = {"steps": []}
    if not work_root.exists():
        return summary

    for step_dir in sorted([p for p in work_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        db_path = step_dir / "queue.db"
        if not db_path.exists():
            continue
        step_name = step_dir.name
        step_info: Dict[str, Any] = {"step": step_name, "db": str(db_path), "claims_deleted": 0, "leader_deleted": 0}
        last_exc: Exception | None = None
        for _attempt in range(10):
            try:
                conn = sqlite3.connect(db_path, timeout=1.0, isolation_level=None)
                conn.row_factory = sqlite3.Row
                try:
                    conn.execute("PRAGMA busy_timeout = 1000")
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        try:
                            row = conn.execute("SELECT COUNT(*) AS c FROM claims").fetchone()
                            step_info["claims_deleted"] = int(row["c"] or 0) if row else 0
                            conn.execute("DELETE FROM claims")
                        except sqlite3.OperationalError:
                            # Table missing or corrupted schema; best-effort.
                            pass
                        try:
                            # leader is a single-row table (id=1); delete unconditionally.
                            row = conn.execute("SELECT COUNT(*) AS c FROM leader").fetchone()
                            step_info["leader_deleted"] = int(row["c"] or 0) if row else 0
                            conn.execute("DELETE FROM leader")
                        except sqlite3.OperationalError:
                            pass
                        conn.execute("COMMIT")
                        last_exc = None
                        break
                    except Exception:
                        try:
                            conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                finally:
                    conn.close()
            except sqlite3.OperationalError as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "locked" in msg or "busy" in msg:
                    time.sleep(0.25)
                    continue
                break
            except Exception as exc:
                last_exc = exc
                break
        if last_exc is not None:
            step_info["error"] = str(last_exc)
        summary["steps"].append(step_info)
    return summary


def load_progress(work_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        wq = WorkQueue.from_step_dir(Path(work_dir))
        if not wq.db_path.exists():
            return None
        progress = wq.progress()
        counts = progress.get("counts") if progress else None
        total = sum(counts.values()) if isinstance(counts, dict) else 0
        leader = wq.leader_status()
        if leader and total == 0:
            status = str(leader.get("status") or "")
            if status in {"running", "completed", "failed"}:
                progress["status"] = status
        return progress
    except Exception:
        return None


def wait_for_step(
    step_dir: Path,
    *,
    timeout: Optional[float] = None,
    poll_seconds: float = 2.0,
) -> WaitResult:
    step_dir = Path(step_dir)
    start = time.time()
    while True:
        if (step_dir / "queue.db").exists():
            wq = WorkQueue.from_step_dir(step_dir)
            counts = wq.counts()
            total = sum(counts.values())
            leader = wq.leader_status() if total == 0 else None
            if leader:
                status = str(leader.get("status") or "")
                if status == "failed":
                    return WaitResult(status="failed", reason=None, failed=True)
                if status == "completed":
                    progress = wq.progress()
                    return WaitResult(
                        status="completed",
                        reason=None,
                        counts=counts,
                        progress=progress,
                        complete=True,
                    )
                if status == "running":
                    # Leader is still active; do not short-circuit on counts.
                    progress = wq.progress()
                    if progress:
                        progress["status"] = "running"
                    if timeout is not None and (time.time() - start) >= float(timeout):
                        return WaitResult(status="timeout", reason="timeout")
                    time.sleep(max(float(poll_seconds), 0.5))
                    continue
            progress = wq.progress()
            counts = progress.get("counts") if progress else None
            if counts and sum(counts.values()) == 0 and leader is None:
                # No items and no leader row yet; treat as running until a leader appears.
                if timeout is not None and (time.time() - start) >= float(timeout):
                    return WaitResult(status="timeout", reason="timeout")
                time.sleep(max(float(poll_seconds), 0.5))
                continue
            if progress and str(progress.get("status") or "") == "completed":
                return WaitResult(
                    status="completed",
                    reason=None,
                    counts=counts,
                    progress=progress,
                    complete=True,
                )
        if timeout is not None and (time.time() - start) >= float(timeout):
            return WaitResult(status="timeout", reason="timeout")
        time.sleep(max(float(poll_seconds), 0.5))
