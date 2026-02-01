from __future__ import annotations

import csv
import os
import sys
import threading
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ..heartbeat import HeartbeatReporter, start_keepalive
from ..io import ensure_dir


@dataclass
class StepContext:
    out_dir: Path
    input_data: dict
    state: dict
    run_id: int
    rank: int
    world_size: int
    local_rank: int
    reuse: bool
    heartbeat: Optional[HeartbeatReporter]
    work_queue: Optional[dict] = None


class StepError(RuntimeError):
    pass


class Step:
    name: str = ""
    stage: str = ""
    supports_indices: bool = True
    supports_work_queue: bool = False
    work_queue_mode: str = "items"  # "items" | "leader"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        cfg_name = cfg.get("name")
        if cfg_name:
            self.name = str(cfg_name)

    def expected_total(self, ctx: StepContext) -> int:
        raise NotImplementedError

    def scan_done(self, ctx: StepContext) -> set[int]:
        return set()

    def run_indices(self, ctx: StepContext, indices: list[int]) -> None:
        raise NotImplementedError

    def run_full(self, ctx: StepContext) -> None:
        # Default to index-based run
        indices = list(range(self.expected_total(ctx)))
        self.run_indices(ctx, indices)

    def write_manifest(self, ctx: StepContext) -> None:
        return

    def build_items(self, ctx: StepContext) -> list[Any]:
        raise NotImplementedError

    def item_done(self, ctx: StepContext, item: Any) -> bool:
        raise NotImplementedError

    def run_item(self, ctx: StepContext, item: Any) -> None:
        raise NotImplementedError

    def run(self, ctx: StepContext) -> None:
        wq_cfg = ctx.work_queue or {}
        if self.supports_work_queue and bool(wq_cfg.get("enabled")):
            mode = str(getattr(self, "work_queue_mode", "items") or "items")
            if mode == "leader":
                return self._run_work_queue_leader(ctx, wq_cfg)
            return self._run_work_queue_items(ctx, wq_cfg)

        expected_total = self.expected_total(ctx)
        done = self.scan_done(ctx)
        missing = [i for i in range(expected_total) if i not in done]
        done_count = len(done)
        missing_count = len(missing)
        output_rows = None
        manifest = self.cfg.get("manifest")
        if manifest:
            p = Path(manifest)
            if not p.is_absolute():
                p = ctx.out_dir / p
            if p.exists():
                try:
                    with p.open("r", newline="") as handle:
                        rows = list(csv.reader(handle))
                    output_rows = max(len(rows) - 1, 0)
                except Exception:
                    output_rows = None
        extra = ""
        if output_rows is not None:
            extra = f" outputs={output_rows}"

        if not missing:
            print(
                f"[{self.name}] reuse_check expected={expected_total} done={done_count} missing=0 run=0{extra}",
                file=sys.__stdout__,
                flush=True,
            )
            try:
                if self.cfg.get("manifest"):
                    self.write_manifest(ctx)
            except Exception:
                pass
            return

        if self.supports_indices:
            my_missing = [i for i in missing if (i % max(ctx.world_size, 1)) == ctx.rank]
            if not my_missing:
                print(
                    f"[{self.name}] reuse_check expected={expected_total} done={done_count} missing={missing_count} run=0{extra}",
                    file=sys.__stdout__,
                    flush=True,
                )
                return
            print(
                f"[{self.name}] reuse_check expected={expected_total} done={done_count} missing={missing_count} run={len(my_missing)}{extra}",
                file=sys.__stdout__,
                flush=True,
            )
            owned_total = ((expected_total - 1 - ctx.rank) // max(ctx.world_size, 1) + 1) if expected_total - 1 >= ctx.rank else 0
            hb = ctx.heartbeat
            if hb:
                hb.start(expected_total=owned_total, primary_counter=self.name)
            keepalive = start_keepalive(hb, extra={"step": self.name})
            try:
                self.run_indices(ctx, my_missing)
            finally:
                if keepalive:
                    stop, thread = keepalive
                    stop.set()
                    thread.join(timeout=1.0)
                if hb:
                    hb.complete(extra={"step": self.name})
            try:
                if self.cfg.get("manifest"):
                    self.write_manifest(ctx)
            except Exception:
                pass
        else:
            # Step does not support partial indices; run full step if anything missing.
            print(
                f"[{self.name}] reuse_check expected={expected_total} done={done_count} missing={missing_count} run=1{extra}",
                file=sys.__stdout__,
                flush=True,
            )
            hb = ctx.heartbeat
            if hb:
                hb.start(expected_total=expected_total, primary_counter=self.name)
            keepalive = start_keepalive(hb, extra={"step": self.name})
            try:
                self.run_full(ctx)
            finally:
                if keepalive:
                    stop, thread = keepalive
                    stop.set()
                    thread.join(timeout=1.0)
                if hb:
                    hb.complete(extra={"step": self.name})
            try:
                if self.cfg.get("manifest"):
                    self.write_manifest(ctx)
            except Exception:
                pass

    def _work_queue_meta(self, ctx: StepContext) -> dict:
        job = (ctx.state or {}).get("job") or {}
        config_path = self.cfg.get("config_path")
        config_sha = None
        if config_path:
            try:
                from ..state import sha256_file

                p = Path(str(config_path))
                if not p.is_absolute():
                    p = ctx.out_dir / p
                if p.exists():
                    config_sha = sha256_file(p)
            except Exception:
                config_sha = None
        return {
            "input_sha256": job.get("input_sha256"),
            "tool_versions": job.get("tool_versions"),
            "config_path": str(config_path) if config_path else None,
            "config_sha256": config_sha,
        }

    def _run_work_queue_items(self, ctx: StepContext, wq_cfg: dict) -> None:
        from ..work_queue import WorkQueue, WorkQueueError

        worker_id = f"{socket.gethostname()}:{os.getpid()}:{ctx.rank}"
        wq = WorkQueue(ctx.out_dir, self.name, wq_cfg, worker_id=worker_id)
        items = self.build_items(ctx)
        try:
            rebuild = bool(wq_cfg.get("rebuild_from_outputs"))
            wq.init_items(
                items,
                self._work_queue_meta(ctx),
                rebuild=rebuild,
                item_done_fn=(lambda item: self.item_done(ctx, item)) if rebuild else None,
            )
        except WorkQueueError as exc:
            raise StepError(str(exc)) from exc

        counts = wq.counts()
        expected_total = sum(counts.values())
        done_count = counts.get("done", 0)
        missing_count = expected_total - done_count

        output_rows = None
        manifest = self.cfg.get("manifest")
        if manifest:
            p = Path(manifest)
            if not p.is_absolute():
                p = ctx.out_dir / p
            if p.exists():
                try:
                    with p.open("r", newline="") as handle:
                        rows = list(csv.reader(handle))
                    output_rows = max(len(rows) - 1, 0)
                except Exception:
                    output_rows = None
        extra = ""
        if output_rows is not None:
            extra = f" outputs={output_rows}"

        allow_failures = bool((ctx.input_data.get("options") or {}).get("continue_on_item_error"))

        if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
            print(
                f"[{self.name}] reuse_check expected={expected_total} done={done_count} missing=0 run=0{extra}",
                file=sys.__stdout__,
                flush=True,
            )
            if not wq.allow_reuse and done_count > 0:
                missing = []
                for work_item, status in wq.iter_items(status="done"):
                    if not self.item_done(ctx, work_item):
                        missing.append(work_item.id)
                        if len(missing) >= 5:
                            break
                if missing:
                    sample = ", ".join(missing)
                    raise StepError(
                        f"{self.name} outputs missing for {len(missing)} done items (e.g., {sample}); "
                        "use --work-queue-rebuild or allow_reuse=true to proceed."
                    )
            if not allow_failures and (counts.get("failed", 0) > 0 or counts.get("blocked", 0) > 0):
                raise StepError(
                    f"{self.name} has failed/blocked items; use --retry-failed or --continue-on-error to proceed."
                )
            self._maybe_write_manifest(ctx, wq)
            return

        print(
            f"[{self.name}] reuse_check expected={expected_total} done={done_count} missing={missing_count} run={counts.get('pending', 0)}{extra}",
            file=sys.__stdout__,
            flush=True,
        )

        prog = wq.progress()

        hb = ctx.heartbeat
        success = False
        if hb:
            hb.start(expected_total=expected_total, primary_counter=self.name)
            hb.update(
                produced_total=int(prog.get("produced_total", 0)),
                expected_total=int(prog.get("expected_total", expected_total)),
                state=str(prog.get("status") or "running"),
            )
        keepalive = start_keepalive(hb, extra={"step": self.name})

        os.environ["PPIFLOW_WORK_QUEUE_DIR"] = str(wq.base_dir)
        os.environ["PPIFLOW_WORK_QUEUE_MODE"] = "items"

        wait_started: float | None = None
        error_reason: str | None = None

        try:
            while True:
                batch: list[Any] = []
                for _ in range(max(int(wq.batch_size), 1)):
                    claimed = wq.claim_next()
                    if not claimed:
                        break
                    batch.append(claimed)
                if not batch:
                    counts = wq.counts()
                    if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
                        break
                    prog = wq.progress()
                    if hb:
                        hb.update(
                            produced_total=int(prog.get("produced_total", 0)),
                            expected_total=int(prog.get("expected_total", expected_total)),
                            state=str(prog.get("status") or "running"),
                        )
                    if wait_started is None:
                        wait_started = time.time()
                    if wq.wait_timeout and (time.time() - wait_started) > float(wq.wait_timeout):
                        error_reason = f"{self.name} wait timeout while items still running"
                        raise StepError(error_reason)
                    time.sleep(max(min(int(wq.lease_seconds) // 6, 10), 2))
                    continue
                wait_started = None
                for claimed in batch:
                    item = claimed.item
                    attempt = claimed.attempt

                    stop_evt = threading.Event()
                    claim_thread = None
                    interval = max(min(int(wq.lease_seconds) // 3, 30), 5)

                    def _claim_loop() -> None:
                        while not stop_evt.wait(interval):
                            try:
                                wq.heartbeat(item.id)
                            except Exception:
                                pass

                    try:
                        claim_thread = threading.Thread(target=_claim_loop, daemon=True)
                        claim_thread.start()
                        # run item with per-item log
                        attempt_log = wq.attempt_log_path(item.id, attempt)
                        prev_log = self.cfg.get("_log_file")
                        prev_prefix = os.environ.get("PPIFLOW_LOG_PREFIX")
                        prev_log_path = os.environ.get("PPIFLOW_LOG_PATH")
                        self.cfg["_log_file"] = str(attempt_log)
                        os.environ["PPIFLOW_LOG_PREFIX"] = f"{item.id}:{attempt}"
                        os.environ["PPIFLOW_LOG_PATH"] = str(attempt_log)
                        try:
                            self.run_item(ctx, item)
                        finally:
                            self.cfg["_log_file"] = prev_log
                            if prev_prefix is None:
                                os.environ.pop("PPIFLOW_LOG_PREFIX", None)
                            else:
                                os.environ["PPIFLOW_LOG_PREFIX"] = prev_prefix
                            if prev_log_path is None:
                                os.environ.pop("PPIFLOW_LOG_PATH", None)
                            else:
                                os.environ["PPIFLOW_LOG_PATH"] = prev_log_path
                        wq.mark_done(item.id, attempt)
                    except FileNotFoundError as exc:
                        wq.mark_blocked(item.id, attempt, str(exc))
                        if not allow_failures:
                            error_reason = str(exc)
                            raise
                    except Exception as exc:
                        wq.mark_failed(item.id, attempt, str(exc))
                        if not allow_failures:
                            error_reason = str(exc)
                            raise
                    finally:
                        stop_evt.set()
                        if claim_thread is not None:
                            claim_thread.join(timeout=1.0)
                    prog = wq.progress()
                    if hb:
                        hb.update(
                            produced_total=int(prog.get("produced_total", 0)),
                            expected_total=int(prog.get("expected_total", expected_total)),
                            state=str(prog.get("status") or "running"),
                        )
            counts = wq.counts()
            if not allow_failures and (counts.get("failed", 0) > 0 or counts.get("blocked", 0) > 0):
                error_reason = (
                    f"{self.name} has failed/blocked items; use --retry-failed or --continue-on-error to proceed."
                )
                raise StepError(error_reason)
            success = True
        finally:
            if keepalive:
                stop, thread = keepalive
                stop.set()
                thread.join(timeout=1.0)
            if hb:
                if success:
                    hb.complete(extra={"step": self.name})
                else:
                    hb.update(
                        produced_total=0,
                        expected_total=int(expected_total),
                        state="failed",
                        force=True,
                    )
            if not success and not allow_failures:
                if not error_reason:
                    error_reason = f"{self.name} work queue failed"
            os.environ.pop("PPIFLOW_WORK_QUEUE_DIR", None)
            os.environ.pop("PPIFLOW_WORK_QUEUE_MODE", None)

        self._maybe_write_manifest(ctx, wq)

    def _run_work_queue_leader(self, ctx: StepContext, wq_cfg: dict) -> None:
        from ..work_queue import WorkQueue, WorkQueueError

        worker_id = f"{socket.gethostname()}:{os.getpid()}:{ctx.rank}"
        wq = WorkQueue(ctx.out_dir, self.name, wq_cfg, worker_id=worker_id)
        try:
            wq.init_leader(self._work_queue_meta(ctx))
        except WorkQueueError as exc:
            raise StepError(str(exc)) from exc

        expected_total = self.expected_total(ctx)
        output_rows = None
        manifest = self.cfg.get("manifest")
        if manifest:
            p = Path(manifest)
            if not p.is_absolute():
                p = ctx.out_dir / p
            if p.exists():
                try:
                    with p.open("r", newline="") as handle:
                        rows = list(csv.reader(handle))
                    output_rows = max(len(rows) - 1, 0)
                except Exception:
                    output_rows = None
        extra = ""
        if output_rows is not None:
            extra = f" outputs={output_rows}"

        leader_status = wq.leader_status()
        if leader_status and str(leader_status.get("status") or "") == "completed":
            print(
                f"[{self.name}] reuse_check expected={expected_total} done={expected_total} missing=0 run=0{extra}",
                file=sys.__stdout__,
                flush=True,
            )
            return

        allow_failures = bool((ctx.input_data.get("options") or {}).get("continue_on_item_error"))
        wait_started: float | None = None

        # Loop until we become leader or see completion.
        while True:
            leader_status = wq.leader_status()
            if leader_status and str(leader_status.get("status") or "") == "completed":
                print(
                    f"[{self.name}] reuse_check expected={expected_total} done={expected_total} missing=0 run=0{extra}",
                    file=sys.__stdout__,
                    flush=True,
                )
                return
            if leader_status and str(leader_status.get("status") or "") == "failed":
                raise StepError(f"{self.name} leader marked failed")
            if not wq.acquire_leader():
                if wait_started is None:
                    wait_started = time.time()
                if wq.wait_timeout and (time.time() - wait_started) > float(wq.wait_timeout):
                    raise StepError(f"{self.name} wait timeout while leader running")
                time.sleep(max(min(int(wq.leader_timeout) // 6, 10), 2))
                continue
            break

        print(
            f"[{self.name}] reuse_check expected={expected_total} done=0 missing={expected_total} run=1{extra}",
            file=sys.__stdout__,
            flush=True,
        )

        hb = ctx.heartbeat
        if hb:
            hb.start(expected_total=expected_total, primary_counter=self.name)
        keepalive = start_keepalive(hb, extra={"step": self.name})

        os.environ["PPIFLOW_WORK_QUEUE_DIR"] = str(wq.base_dir)
        os.environ["PPIFLOW_WORK_QUEUE_MODE"] = "leader"

        leader_stop = threading.Event()
        leader_interval = max(min(int(wq.leader_timeout) // 3, 30), 5)

        def _leader_loop() -> None:
            while not leader_stop.wait(leader_interval):
                try:
                    wq.leader_heartbeat()
                except Exception:
                    pass

        leader_thread = threading.Thread(target=_leader_loop, daemon=True)
        leader_thread.start()

        success = False
        error_reason: str | None = None
        try:
            try:
                self.run_full(ctx)
                try:
                    if self.cfg.get("manifest"):
                        self.write_manifest(ctx)
                except Exception:
                    pass
                wq.write_complete()
                success = True
            except Exception as exc:
                error_reason = str(exc)
                wq.write_failed(error=error_reason)
                raise
        finally:
            leader_stop.set()
            leader_thread.join(timeout=1.0)
            if keepalive:
                stop, thread = keepalive
                stop.set()
                thread.join(timeout=1.0)
            if hb:
                if success:
                    hb.complete(extra={"step": self.name})
                else:
                    hb.update(
                        produced_total=0,
                        expected_total=int(expected_total),
                        state="failed",
                        force=True,
                    )
            wq.release_leader()
            os.environ.pop("PPIFLOW_WORK_QUEUE_DIR", None)
            os.environ.pop("PPIFLOW_WORK_QUEUE_MODE", None)

    def _maybe_write_manifest(self, ctx: StepContext, wq) -> None:
        if not self.cfg.get("manifest"):
            return
        try:
            counts = wq.counts()
        except Exception:
            return
        if counts.get("pending", 0) != 0 or counts.get("running", 0) != 0:
            return
        if not wq.acquire_leader():
            return
        try:
            self.write_manifest(ctx)
        except Exception:
            pass
        finally:
            wq.release_leader()

    def output_dir(self, ctx: StepContext) -> Path:
        out_dir = self.cfg.get("output_dir")
        if not out_dir:
            raise StepError(f"Step {self.name} missing output_dir in config")
        p = Path(out_dir)
        if not p.is_absolute():
            p = ctx.out_dir / p
        ensure_dir(p)
        return p

    def manifest_path(self, ctx: StepContext) -> Path:
        manifest = self.cfg.get("manifest")
        if not manifest:
            raise StepError(f"Step {self.name} missing manifest path in config")
        p = Path(manifest)
        if not p.is_absolute():
            p = ctx.out_dir / p
        ensure_dir(p.parent)
        return p
