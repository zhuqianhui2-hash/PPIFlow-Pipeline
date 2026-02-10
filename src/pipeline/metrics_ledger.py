from __future__ import annotations

import json
import os
import sqlite3
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .run_lock import RunLockError, ensure_expected_lock_id, run_lock_disabled, validate_expected_lock_id


def _iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _jsonable(value: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable Python primitives.

    We keep this intentionally small and dependency-free (no numpy/torch imports).
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # numpy / torch scalars often define item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    # Fallback: string repr (still useful for debugging).
    return str(value)


@dataclass
class LedgerRow:
    item_id: str
    status: str
    attempt: int
    worker_id: str | None
    updated_at: str | None
    error: str | None
    design_id: int | None
    structure_id: str | None
    outputs: dict[str, Any]
    metrics: dict[str, Any]


class MetricsLedger:
    """
    Per-step metrics ledger.

    Canonical store: <step_output_dir>/metrics.db (SQLite, WAL).
    Derived exports (CSV) are produced by a single writer during finalize.
    """

    def __init__(
        self,
        run_dir: str | Path,
        step_output_dir: str | Path,
        *,
        sqlite_journal_mode: str = "WAL",
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.step_output_dir = Path(step_output_dir)
        self.db_path = self.step_output_dir / "metrics.db"
        self.sqlite_journal_mode = str(sqlite_journal_mode or "WAL").upper()
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._conn: sqlite3.Connection | None = None
        # Snapshot expected lock id if provided; otherwise resolve lazily on first write.
        self._expected_run_lock_id: str | None = os.environ.get("PPIFLOW_RUN_LOCK_ID")

    @staticmethod
    def default_worker_id() -> str:
        rank = os.environ.get("RANK", "0")
        return f"{socket.gethostname()}:{os.getpid()}:{rank}"

    def _validate_run_lock(self) -> None:
        if run_lock_disabled():
            return
        expected = self._expected_run_lock_id
        if not expected:
            try:
                expected = ensure_expected_lock_id(self.run_dir, wait_seconds=5.0)
            except RunLockError as exc:
                raise RuntimeError(str(exc)) from exc
            self._expected_run_lock_id = expected
        validate_expected_lock_id(self.run_dir, expected_lock_id=expected)

    def _connect(self) -> sqlite3.Connection:
        self.step_output_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA journal_mode = {self.sqlite_journal_mode}")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        return conn

    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._connect()
            self._ensure_schema(self._conn)
        return self._conn

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        finally:
            self._conn = None

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
              item_id      TEXT PRIMARY KEY,
              design_id    INTEGER,
              structure_id TEXT,
              status       TEXT,
              attempt      INTEGER,
              worker_id    TEXT,
              error        TEXT,
              updated_at   TEXT,
              outputs_json TEXT,
              metrics_json TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_status ON metrics(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_updated_at ON metrics(updated_at)")

    def upsert(
        self,
        item_id: str,
        *,
        status: str,
        metrics: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        attempt: int | None = None,
        worker_id: str | None = None,
        error: str | None = None,
        design_id: int | None = None,
        structure_id: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        self._validate_run_lock()
        item_id = str(item_id)
        now = updated_at or _iso()
        metrics_payload = _jsonable(metrics or {})
        outputs_payload = _jsonable(outputs or {})
        try:
            metrics_json = json.dumps(metrics_payload, separators=(",", ":"))
        except Exception:
            metrics_json = json.dumps({"__unserializable__": str(metrics_payload)})
        try:
            outputs_json = json.dumps(outputs_payload, separators=(",", ":"))
        except Exception:
            outputs_json = json.dumps({"__unserializable__": str(outputs_payload)})

        if attempt is None:
            try:
                attempt = int(os.environ.get("PPIFLOW_ATTEMPT") or 1)
            except Exception:
                attempt = 1
        attempt_i = int(attempt)

        conn = self.conn()
        # Rely on busy_timeout, but add a short retry loop for transient "database is locked".
        deadline = time.time() + 30.0
        while True:
            try:
                conn.execute(
                    """
                    INSERT INTO metrics(
                        item_id, design_id, structure_id, status, attempt,
                        worker_id, error, updated_at, outputs_json, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        design_id=excluded.design_id,
                        structure_id=excluded.structure_id,
                        status=excluded.status,
                        attempt=excluded.attempt,
                        worker_id=excluded.worker_id,
                        error=excluded.error,
                        updated_at=excluded.updated_at,
                        outputs_json=excluded.outputs_json,
                        metrics_json=excluded.metrics_json
                    """,
                    (
                        item_id,
                        design_id,
                        structure_id,
                        str(status),
                        attempt_i,
                        worker_id,
                        error,
                        now,
                        outputs_json,
                        metrics_json,
                    ),
                )
                return
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" in msg and time.time() < deadline:
                    time.sleep(0.1)
                    continue
                raise

    def _decode_row(self, row: sqlite3.Row) -> LedgerRow:
        outputs: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        try:
            outputs = json.loads(row["outputs_json"] or "{}") or {}
        except Exception:
            outputs = {}
        try:
            metrics = json.loads(row["metrics_json"] or "{}") or {}
        except Exception:
            metrics = {}
        try:
            design_id = int(row["design_id"]) if row["design_id"] is not None else None
        except Exception:
            design_id = None
        return LedgerRow(
            item_id=str(row["item_id"]),
            status=str(row["status"] or ""),
            attempt=int(row["attempt"] or 0),
            worker_id=str(row["worker_id"]) if row["worker_id"] is not None else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
            error=str(row["error"]) if row["error"] is not None else None,
            design_id=design_id,
            structure_id=str(row["structure_id"]) if row["structure_id"] is not None else None,
            outputs=outputs if isinstance(outputs, dict) else {},
            metrics=metrics if isinstance(metrics, dict) else {},
        )

    def get(self, item_id: str) -> LedgerRow | None:
        conn = self.conn()
        row = conn.execute(
            "SELECT * FROM metrics WHERE item_id = ?",
            (str(item_id),),
        ).fetchone()
        if row is None:
            return None
        return self._decode_row(row)

    def iter_rows(self, *, status: str | None = None) -> Iterable[LedgerRow]:
        conn = self.conn()
        if status is None:
            cur = conn.execute("SELECT * FROM metrics")
        else:
            cur = conn.execute("SELECT * FROM metrics WHERE status = ?", (str(status),))
        for row in cur.fetchall():
            yield self._decode_row(row)

    def has_done(self, item_id: str) -> bool:
        row = self.get(item_id)
        return row is not None and str(row.status) == "done"

    def checkpoint_and_truncate_wal(self) -> None:
        """
        Ensure the DB is self-contained (avoid losing data when copying outputs without WAL sidecars).
        Leader-only.
        """
        conn = self.conn()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            # Best-effort; callers can still copy WAL sidecars if needed.
            pass

    def export_csv(
        self,
        path: str | Path,
        *,
        status: str | None = "done",
        include_item_id: bool = True,
    ) -> None:
        """
        Export a flat CSV view of the ledger.

        This is intentionally generic; step-specific exports may still be produced by step code.
        """
        rows = []
        for r in self.iter_rows(status=status if status else None):
            merged = {}
            if include_item_id:
                merged["item_id"] = r.item_id
            if isinstance(r.metrics, dict):
                merged.update(r.metrics)
            if isinstance(r.outputs, dict):
                # Avoid clobbering metrics keys; outputs are usually paths.
                for k, v in r.outputs.items():
                    merged.setdefault(str(k), v)
            rows.append(merged)
        if not rows:
            return
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.parent / f"{out_path.name}.tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, out_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to export ledger CSV to {path}: {exc}") from exc
