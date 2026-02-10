from __future__ import annotations

import json
import os
import sys
import subprocess
import time
import socket
from pathlib import Path
from typing import Any

from Bio import PDB

from .base import Step, StepContext, StepError
from ..metrics_ledger import MetricsLedger
from ..run_lock import (
    ensure_expected_lock_id as _ensure_run_lock_id,
    validate_expected_lock_id as _validate_run_lock_id,
    read_active_lock_id as _read_active_run_lock_id,
)
from ..work_queue import WorkItem
from ..logging_utils import log_command_progress, run_command
from ..io import ensure_dir, repo_root, write_json
from ..manifests import extract_design_id, structure_id_from_name, write_csv


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p


def _resolve_ctx_path(ctx: StepContext, value: str | Path | None) -> Path | None:
    if not value:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (ctx.out_dir / p).resolve()


def _load_chain_offset_map(offsets_path: Path) -> list[int] | None:
    try:
        payload = json.loads(offsets_path.read_text())
    except Exception:
        return None
    chains = payload.get("chains") if isinstance(payload, dict) else None
    if not chains:
        return None
    mapping: list[int] = []
    for seg in chains:
        try:
            length = int(seg.get("length") or 0)
            start = int(seg.get("start_resseq_B") or 0)
        except Exception:
            return None
        if length <= 0 or start <= 0:
            return None
        for i in range(length):
            mapping.append(start + i)
    return mapping or None


def _renumber_chain_with_offsets(pdb_path: Path, chain_id: str, mapping: list[int]) -> bool:
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("model", str(pdb_path))
        model = structure[0]
        if chain_id not in model:
            return False
        chain = model[chain_id]
        residues = [r for r in chain if r.id[0] == " "]
        if len(residues) != len(mapping):
            return False
        for idx, res in enumerate(residues, start=1):
            res.id = (" ", int(mapping[idx - 1]), " ")
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path))
        return True
    except Exception:
        return False


class GenStep(Step):
    name = "gen"
    stage = "gen"
    supports_indices = True
    supports_work_queue = True
    work_queue_mode = "items"
    # Drain per-worker batches to avoid repeated model reloads.
    per_worker_batch = True
    # 0 means "claim all available items for this worker".
    batch_size = 0

    def expected_total(self, ctx: StepContext) -> int:
        sampling = ctx.input_data.get("sampling") or {}
        n = int(sampling.get("samples_per_target", 0) or 0)
        if n <= 0:
            raise StepError("sampling.samples_per_target must be > 0")
        return n

    def _parse_existing(self, out_dir: Path, name: str) -> set[int]:
        done: set[int] = set()
        if not out_dir.exists():
            return done
        for fp in out_dir.glob(f"{name}_*.pdb"):
            try:
                stem = fp.stem
                if not stem.startswith(f"{name}_"):
                    continue
                suffix = stem[len(name) + 1 :]
                if suffix.isdigit():
                    done.add(int(suffix))
            except Exception:
                continue
        return done

    def scan_done(self, ctx: StepContext) -> set[int]:
        out_dir = self.output_dir(ctx)
        name = ctx.input_data.get("name", "")
        done = self._parse_existing(out_dir, name)
        expected = self.expected_total(ctx)
        if done and len(done) == expected:
            return done
        # Fallback: include non-numeric stems when count matches expected.
        if expected > 0:
            pdbs = sorted(out_dir.glob(f"{name}_*.pdb"))
            if len(pdbs) == expected:
                return set(range(expected))
        return done

    def _length_schedule_path(self, ctx: StepContext, out_dir: Path) -> Path:
        path = out_dir / "length_schedule.json"
        return path

    def _ensure_length_schedule(self, ctx: StepContext, out_dir: Path) -> Path | None:
        protocol = ctx.input_data.get("protocol")
        if protocol != "binder":
            return None
        binder = ctx.input_data.get("binder") or {}
        length_spec = binder.get("length")
        if not length_spec:
            return None
        if "-" in str(length_spec):
            parts = str(length_spec).split("-")
            min_len, max_len = int(parts[0]), int(parts[1])
        else:
            min_len = max_len = int(length_spec)
        schedule_path = self._length_schedule_path(ctx, out_dir)
        if schedule_path.exists():
            return schedule_path
        # Generate deterministic schedule
        sampling = ctx.input_data.get("sampling") or {}
        n = int(sampling.get("samples_per_target", 0) or 0)
        seed = int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0)
        try:
            import numpy as np

            rng = np.random.default_rng(seed=seed)
            lengths = rng.integers(low=min_len, high=max_len, size=(n,))
            schedule = [int(x) for x in lengths]
        except Exception:
            schedule = [int(min_len)] * n
        write_json(schedule_path, {"lengths": schedule}, indent=2)
        return schedule_path

    def _serialize_sample_ids(self, indices: list[int]) -> str:
        if not indices:
            return ""
        ids = sorted(set(int(i) for i in indices))
        ranges: list[tuple[int, int]] = []
        start = prev = ids[0]
        for val in ids[1:]:
            if val == prev + 1:
                prev = val
                continue
            ranges.append((start, prev))
            start = prev = val
        ranges.append((start, prev))
        parts = [f"{s}-{e}" if s != e else str(s) for s, e in ranges]
        return ",".join(parts)

    def _worker_id(self) -> str:
        rank = os.environ.get("RANK", "0")
        return f"{socket.gethostname()}:{os.getpid()}:{rank}"

    def _input_dir(self, out_dir: Path) -> Path:
        return out_dir / "input"

    def _preprocess_done_path(self, out_dir: Path) -> Path:
        return self._input_dir(out_dir) / "preprocess.done"

    def _preprocess_lock_path(self, out_dir: Path) -> Path:
        return self._input_dir(out_dir) / "preprocess.lock"

    def _preprocessed_csv_path(self, out_dir: Path, name: str) -> Path:
        return self._input_dir(out_dir) / f"{name}_input.csv"

    def _acquire_preprocess_lock(self, out_dir: Path) -> bool:
        lock_path = self._preprocess_lock_path(out_dir)
        ensure_dir(lock_path.parent)
        now = time.time()
        payload = f"{now}|{self._worker_id()}"
        try:
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        return True

    def _preprocess_lock_age(self, out_dir: Path) -> float | None:
        lock_path = self._preprocess_lock_path(out_dir)
        now = time.time()
        try:
            text = lock_path.read_text().strip()
            if text:
                ts_raw = text.split("|", 1)[0]
                ts = float(ts_raw)
                return max(0.0, now - ts)
        except Exception:
            pass
        try:
            return max(0.0, now - lock_path.stat().st_mtime)
        except Exception:
            return None

    def _steal_stale_preprocess_lock(self, out_dir: Path, *, max_age_s: float) -> bool:
        lock_path = self._preprocess_lock_path(out_dir)
        age = self._preprocess_lock_age(out_dir)
        if age is None or age <= max_age_s:
            return False
        ts = f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.{time.time_ns()}"
        stale = lock_path.parent / f"preprocess.lock.stale.{ts}"
        try:
            lock_path.rename(stale)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _write_preprocess_done(self, out_dir: Path, *, input_csv: Path) -> None:
        done_path = self._preprocess_done_path(out_dir)
        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "worker_id": self._worker_id(),
            "input_csv": str(input_csv),
        }
        tmp = done_path.parent / f"{done_path.name}.tmp"
        tmp.write_text(json.dumps(payload, separators=(",", ":")))
        os.replace(tmp, done_path)

    def _read_preprocess_done(self, out_dir: Path) -> dict[str, Any] | None:
        path = self._preprocess_done_path(out_dir)
        try:
            if not path.exists():
                return None
            payload = json.loads(path.read_text())
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _is_archived_run_lock_id(self, out_dir: Path, lock_id: str) -> bool:
        """
        Return True if this output directory contains an archived run lock with the given id.

        When a controller steals/takes over a run lock, the previous `.ppiflow_lock/` directory is
        renamed to `.ppiflow_lock.stale.<timestamp>.<lock_id>`. Accepting preprocess markers from
        an archived lock id lets us resume without forcing a full re-preprocess after takeover.
        """
        lock_id = str(lock_id or "").strip()
        if not lock_id:
            return False
        try:
            for p in out_dir.glob(f".ppiflow_lock.stale.*.{lock_id}"):
                if p.exists():
                    return True
        except Exception:
            return False
        return False

    def _preprocess_done_valid(self, ctx: StepContext, out_dir: Path, *, input_csv: Path) -> bool:
        payload = self._read_preprocess_done(out_dir)
        if not payload:
            return False
        try:
            if str(payload.get("input_csv") or "") != str(input_csv):
                return False
        except Exception:
            return False
        # Require strong metadata so reuse/resume is safe across restarts/world-size changes.
        if payload.get("expected_total") is None:
            return False
        if payload.get("protocol") is None:
            return False
        if payload.get("name") is None:
            return False
        # Run-lock fencing:
        # - Prefer markers from the currently active controller lock_id.
        # - Allow markers from a *previous* controller if that lock_id has been archived in this out_dir
        #   (i.e., lock steal/takeover), so resume doesn't force re-preprocess.
        # - If a run lock exists but owner.json is temporarily missing/unreadable, treat markers as invalid
        #   until the lock becomes readable.
        active_lock_id = _read_active_run_lock_id(ctx.out_dir)
        if (ctx.out_dir / ".ppiflow_lock").exists() and active_lock_id is None:
            return False
        marker_lock_id = payload.get("run_lock_id")
        if active_lock_id is not None:
            if marker_lock_id is None:
                return False
            if str(marker_lock_id or "") != str(active_lock_id):
                if not self._is_archived_run_lock_id(ctx.out_dir, str(marker_lock_id or "")):
                    return False
        try:
            expected_total = int(payload.get("expected_total") or 0)
        except Exception:
            expected_total = 0
        if expected_total <= 0 or expected_total != int(self.expected_total(ctx)):
            return False
        if payload.get("protocol") and str(payload.get("protocol")) != str(ctx.input_data.get("protocol") or ""):
            return False
        expected_name = str(ctx.input_data.get("name") or "design")
        if payload.get("name") and str(payload.get("name")) != expected_name:
            return False
        return True

    def _ensure_preprocessed(self, ctx: StepContext, out_dir: Path) -> Path | None:
        """
        Leader-only preprocessing for gen entrypoints (binder + antibody/VHH).

        In work_queue_mode="items", multiple ranks/workers may start in arbitrary order.
        We coordinate via preprocess.lock + preprocess.done markers under output/input/.
        """
        name = str(ctx.input_data.get("name") or "design")
        input_csv = self._preprocessed_csv_path(out_dir, name)
        done_path = self._preprocess_done_path(out_dir)
        if done_path.exists() and input_csv.exists() and self._preprocess_done_valid(ctx, out_dir, input_csv=input_csv):
            return input_csv

        # Stale-lock policy (seconds).
        try:
            max_age_s = float((ctx.work_queue or {}).get("leader_timeout") or 600)
        except Exception:
            max_age_s = 600.0

        protocol = str(ctx.input_data.get("protocol") or "")
        if protocol not in {"binder", "antibody", "vhh"}:
            return None

        # Run-lock fencing: ensure we are operating under the active controller lock.
        try:
            expected_lock_id = _ensure_run_lock_id(ctx.out_dir, wait_seconds=5.0)
            _validate_run_lock_id(ctx.out_dir, expected_lock_id=expected_lock_id)
        except Exception as exc:
            raise StepError(f"gen preprocessing fenced by run lock failed: {exc}") from exc

        # Try to become the preprocessor.
        if self._acquire_preprocess_lock(out_dir):
            try:
                # Run the entrypoint in preprocess-only mode to generate input CSV + PKLs.
                tools = ctx.input_data.get("tools") or {}
                config_path = tools.get("ppiflow_config")

                env = os.environ.copy()
                env["RANK"] = "0"
                env["LOCAL_RANK"] = "0"
                env["WORLD_SIZE"] = "1"

                if protocol == "binder":
                    script = _resolve_repo_path("src/entrypoints/sample_binder.py")
                    config_default = tools.get("ppiflow_binder_config") or str(
                        _resolve_repo_path("src/configs/inference_binder.yaml")
                    )
                    config_path = config_path or config_default
                    config_path = str(_resolve_repo_path(config_path))
                    binder = ctx.input_data.get("binder") or {}
                    length_spec = binder.get("length")
                    if not length_spec:
                        raise StepError("binder.length missing")
                    if "-" in str(length_spec):
                        min_len, max_len = [int(x) for x in str(length_spec).split("-")]
                    else:
                        min_len = max_len = int(length_spec)
                    target = ctx.input_data.get("target") or {}
                    target_chains = target.get("chains") or []
                    target_chain = target_chains[0] if target_chains else None
                    binder_chain = ctx.input_data.get("binder_chain") or "A"
                    hotspots = target.get("hotspots")
                    hotspots_file = target.get("hotspots_file")
                    ckpt = tools.get("ppiflow_ckpt")
                    if not ckpt:
                        raise StepError("tools.ppiflow_ckpt is required for binder generation")

                    cmd = [
                        sys.executable,
                        str(script),
                        "--input_pdb",
                        str(target.get("pdb")),
                        "--target_chain",
                        str(target_chain),
                        "--binder_chain",
                        str(binder_chain),
                        "--config",
                        str(config_path),
                        "--samples_min_length",
                        str(min_len),
                        "--samples_max_length",
                        str(max_len),
                        "--samples_per_target",
                        str(self.expected_total(ctx)),
                        "--model_weights",
                        str(ckpt),
                        "--output_dir",
                        str(out_dir),
                        "--name",
                        str(name),
                        "--seed",
                        str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0)),
                        "--preprocess_only",
                    ]
                    if hotspots_file:
                        cmd.extend(["--hotspots_file", str(hotspots_file)])
                    elif hotspots:
                        if isinstance(hotspots, list):
                            cmd.extend(["--specified_hotspots", ",".join(hotspots)])
                        else:
                            cmd.extend(["--specified_hotspots", str(hotspots)])
                else:
                    script = _resolve_repo_path("src/entrypoints/sample_antibody_nanobody.py")
                    config_default = tools.get("ppiflow_antibody_config") or str(
                        _resolve_repo_path("src/configs/inference_nanobody.yaml")
                    )
                    config_path = config_path or config_default
                    config_path = str(_resolve_repo_path(config_path))
                    framework = ctx.input_data.get("framework") or {}
                    ckpt = tools.get("ppiflow_ckpt")
                    if not ckpt:
                        raise StepError("tools.ppiflow_ckpt is required for antibody/vhh generation")
                    target = ctx.input_data.get("target") or {}
                    cmd = [
                        sys.executable,
                        str(script),
                        "--antigen_pdb",
                        str(target.get("pdb")),
                        "--framework_pdb",
                        str(framework.get("pdb")),
                        "--antigen_chain",
                        ",".join(target.get("chains") or []),
                        "--heavy_chain",
                        str(framework.get("heavy_chain")),
                        "--cdr_length",
                        str(framework.get("cdr_length")),
                        "--samples_per_target",
                        str(self.expected_total(ctx)),
                        "--config",
                        str(config_path),
                        "--model_weights",
                        str(ckpt),
                        "--output_dir",
                        str(out_dir),
                        "--name",
                        str(name),
                        "--seed",
                        str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0)),
                        "--preprocess_only",
                    ]
                    light_chain = framework.get("light_chain")
                    if light_chain:
                        cmd.extend(["--light_chain", str(light_chain)])
                    hotspots = target.get("hotspots")
                    hotspots_file = target.get("hotspots_file")
                    if hotspots_file:
                        cmd.extend(["--hotspots_file", str(hotspots_file)])
                    elif hotspots:
                        if isinstance(hotspots, list):
                            cmd.extend(["--specified_hotspots", ",".join(hotspots)])
                        else:
                            cmd.extend(["--specified_hotspots", str(hotspots)])

                run_command(
                    cmd,
                    env=env,
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )

                if not input_csv.exists():
                    raise StepError(f"Preprocess did not produce expected input CSV: {input_csv}")
                # Record a strong marker so reuse/resume is config-world-size agnostic.
                # Validate again before writing the marker so a straggler can't stamp a marker for a new lock.
                _validate_run_lock_id(ctx.out_dir, expected_lock_id=expected_lock_id)

                done_payload = {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "worker_id": self._worker_id(),
                    "input_csv": str(input_csv),
                    "expected_total": int(self.expected_total(ctx)),
                    "protocol": str(ctx.input_data.get("protocol") or ""),
                    "name": str(name),
                    "run_lock_id": expected_lock_id,
                }
                tmp = done_path.parent / f"{done_path.name}.tmp"
                tmp.write_text(json.dumps(done_payload, separators=(",", ":")))
                os.replace(tmp, done_path)
            finally:
                try:
                    self._preprocess_lock_path(out_dir).unlink()
                except Exception:
                    pass
            return input_csv

        # Wait for preprocessing to complete (or steal stale lock).
        deadline = time.time() + max(max_age_s, 60.0)
        while time.time() < deadline:
            if done_path.exists() and input_csv.exists() and self._preprocess_done_valid(ctx, out_dir, input_csv=input_csv):
                return input_csv
            self._steal_stale_preprocess_lock(out_dir, max_age_s=max_age_s)
            if self._acquire_preprocess_lock(out_dir):
                # Loop back; we acquired after stealing.
                return self._ensure_preprocessed(ctx, out_dir)
            time.sleep(1.0)
        raise StepError(f"Timed out waiting for gen preprocessing: {done_path}")

    def outputs_complete(self, ctx: StepContext) -> bool:
        if not super().outputs_complete(ctx):
            return False
        out_dir = self._resolve_output_dir_path(ctx)
        if not (out_dir / "metrics.db").exists():
            if not self._allow_legacy_outputs(ctx):
                return False
        if not (out_dir / "sample_metrics.csv").exists():
            return False
        return True

    def run_indices(self, ctx: StepContext, indices: list[int]) -> None:
        protocol = ctx.input_data.get("protocol")
        out_dir = self.output_dir(ctx)
        name = ctx.input_data.get("name")
        tools = ctx.input_data.get("tools") or {}
        config_path = tools.get("ppiflow_config")
        sample_ids = self._serialize_sample_ids(indices)
        target = ctx.input_data.get("target") or {}
        target_chain = (target.get("chains") or [None])[0]
        offsets_path = _resolve_ctx_path(ctx, target.get("chain_offsets"))
        offset_map = None
        if offsets_path:
            if not offsets_path.exists():
                raise StepError(f"Target offsets file not found: {offsets_path}")
            offset_map = _load_chain_offset_map(offsets_path)
            if not offset_map:
                raise StepError(f"Failed to load target chain offsets: {offsets_path}")

        if protocol == "binder":
            script = _resolve_repo_path("src/entrypoints/sample_binder.py")
            config_default = tools.get("ppiflow_binder_config") or str(
                _resolve_repo_path("src/configs/inference_binder.yaml")
            )
            config_path = config_path or config_default
            config_path = str(_resolve_repo_path(config_path))
            binder = ctx.input_data.get("binder") or {}
            length_spec = binder.get("length")
            if not length_spec:
                raise StepError("binder.length missing")
            if "-" in str(length_spec):
                min_len, max_len = [int(x) for x in str(length_spec).split("-")]
            else:
                min_len = max_len = int(length_spec)

            target_chains = target.get("chains") or []
            target_chain = target_chains[0] if target_chains else None
            binder_chain = ctx.input_data.get("binder_chain") or "A"
            hotspots = target.get("hotspots")
            hotspots_file = target.get("hotspots_file")
            ckpt = tools.get("ppiflow_ckpt")
            if not ckpt:
                raise StepError("tools.ppiflow_ckpt is required for binder generation")

            preprocess_csv = self._ensure_preprocessed(ctx, out_dir)
            cmd = [
                sys.executable,
                str(script),
                "--input_csv",
                str(preprocess_csv),
                "--target_chain",
                str(target_chain),
                "--binder_chain",
                str(binder_chain),
                "--config",
                str(config_path),
                "--samples_min_length",
                str(min_len),
                "--samples_max_length",
                str(max_len),
                "--samples_per_target",
                str(self.expected_total(ctx)),
                "--model_weights",
                str(ckpt),
                "--output_dir",
                str(out_dir),
                "--name",
                str(name),
                "--seed",
                str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0)),
                "--sample_ids",
                sample_ids,
            ]
            if hotspots_file:
                cmd.extend(["--hotspots_file", str(hotspots_file)])
            elif hotspots:
                if isinstance(hotspots, list):
                    cmd.extend(["--specified_hotspots", ",".join(hotspots)])
                else:
                    cmd.extend(["--specified_hotspots", str(hotspots)])
            schedule_path = self._ensure_length_schedule(ctx, out_dir)
            if schedule_path is not None:
                cmd.extend(["--length_schedule_path", str(schedule_path)])
        else:
            script = _resolve_repo_path("src/entrypoints/sample_antibody_nanobody.py")
            config_default = tools.get("ppiflow_antibody_config") or str(
                _resolve_repo_path("src/configs/inference_nanobody.yaml")
            )
            config_path = config_path or config_default
            config_path = str(_resolve_repo_path(config_path))
            framework = ctx.input_data.get("framework") or {}
            ckpt = tools.get("ppiflow_ckpt")
            if not ckpt:
                raise StepError("tools.ppiflow_ckpt is required for antibody/vhh generation")
            preprocess_csv = self._ensure_preprocessed(ctx, out_dir)
            cmd = [
                sys.executable,
                str(script),
                "--antigen_pdb",
                str(target.get("pdb")),
                "--framework_pdb",
                str(framework.get("pdb")),
                "--antigen_chain",
                ",".join(target.get("chains") or []),
                "--heavy_chain",
                str(framework.get("heavy_chain")),
                "--cdr_length",
                str(framework.get("cdr_length")),
                "--samples_per_target",
                str(self.expected_total(ctx)),
                "--config",
                str(config_path),
                "--model_weights",
                str(ckpt),
                "--output_dir",
                str(out_dir),
                "--name",
                str(name),
                "--seed",
                str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0)),
                "--sample_ids",
                sample_ids,
                "--skip_preprocess",
                "--input_csv",
                str(preprocess_csv),
            ]
            light_chain = framework.get("light_chain")
            if light_chain:
                cmd.extend(["--light_chain", str(light_chain)])
            hotspots = target.get("hotspots")
            hotspots_file = target.get("hotspots_file")
            if hotspots_file:
                cmd.extend(["--hotspots_file", str(hotspots_file)])
            elif hotspots:
                if isinstance(hotspots, list):
                    cmd.extend(["--specified_hotspots", ",".join(hotspots)])
                else:
                    cmd.extend(["--specified_hotspots", str(hotspots)])

        env = os.environ.copy()
        # Metrics ledger wiring for gen: write per-item metrics incrementally, export CSV on finalize.
        env["PPIFLOW_METRICS_RUN_DIR"] = str(ctx.out_dir)
        env["PPIFLOW_METRICS_STEP_DIR"] = str(out_dir)
        start = time.time()
        status = "OK"
        try:
            run_command(
                cmd,
                env=env,
                log_file=self.cfg.get("_log_file"),
                verbose=bool(self.cfg.get("_verbose")),
            )
            if offset_map and target_chain:
                for sample_id in indices:
                    if protocol == "binder":
                        out_path = out_dir / f"{name}_{sample_id}.pdb"
                    else:
                        out_path = out_dir / f"{name}_antigen_antibody_sample_{sample_id}.pdb"
                    if not out_path.exists():
                        continue
                    if not _renumber_chain_with_offsets(out_path, str(target_chain), offset_map):
                        raise StepError(f"Failed to apply target chain offsets to {out_path}")
        except Exception:
            status = "FAILED"
            raise
        finally:
            log_command_progress(
                str(self.cfg.get("name") or self.name),
                1,
                1,
                item="command",
                status=status,
                elapsed=time.time() - start,
                log_file=self.cfg.get("_log_file"),
            )

    def run_batch(self, ctx: StepContext, items: list[WorkItem]) -> dict[str, tuple[str, str | None]]:
        sample_ids = []
        for item in items:
            try:
                sample_ids.append(int((item.payload or {}).get("sample_id")))
            except Exception:
                sample_ids.append(int(item.id))
        err: str | None = None
        try:
            # Run all sample IDs in a single command.
            self.run_indices(ctx, sample_ids)
        except Exception as exc:
            err = str(exc)
        results: dict[str, tuple[str, str | None]] = {}
        for item in items:
            if self.item_done(ctx, item):
                results[item.id] = ("done", None)
            else:
                results[item.id] = ("failed", err or "missing output")
        return results

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        name = ctx.input_data.get("name", "design")
        rows = []
        for fp in sorted(out_dir.glob(f"{name}_*.pdb")):
            did = extract_design_id(fp.stem)
            rows.append({
                "design_id": did,
                "structure_id": structure_id_from_name(fp.stem),
                "pdb_path": str(fp),
                "parent_id": None,
            })
        if not rows:
            return
        manifest = self.manifest_path(ctx)
        write_csv(manifest, rows, ["design_id", "structure_id", "pdb_path", "parent_id"])

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        total = self.expected_total(ctx)
        items = []
        for i in range(total):
            items.append(WorkItem(id=str(i), payload={"sample_id": i}))
        return items

    def item_done(self, ctx: StepContext, item: WorkItem) -> bool:
        out_dir = self.output_dir(ctx)
        name = ctx.input_data.get("name", "")
        sample_id = int((item.payload or {}).get("sample_id"))
        target = out_dir / f"{name}_{sample_id}.pdb"
        # gen is only "done" when both the output structure exists and metrics are recorded.
        # This makes resume robust against crash windows where PDBs are produced but CSV metrics
        # were never written/flushed.
        ledger = MetricsLedger(ctx.out_dir, out_dir)
        try:
            has_metrics = ledger.has_done(str(sample_id))
        finally:
            ledger.close()
        if target.exists() and has_metrics:
            return True
        # Antibody/VHH outputs use a different naming pattern.
        protocol = ctx.input_data.get("protocol")
        if protocol in {"antibody", "vhh"}:
            alt = out_dir / f"{name}_antigen_antibody_sample_{sample_id}.pdb"
            if alt.exists() and has_metrics:
                return True
        return False

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        sample_id = int((item.payload or {}).get("sample_id"))
        self.run_indices(ctx, [sample_id])

    def _export_sample_metrics_csv(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        ledger = MetricsLedger(ctx.out_dir, out_dir)
        try:
            # Reconcile: if output PDBs exist but the ledger is missing rows (crash window),
            # upsert a minimal row so exports and downstream filters are consistent.
            name = str(ctx.input_data.get("name") or "design")
            protocol = str(ctx.input_data.get("protocol") or "")
            framework = ctx.input_data.get("framework") or {}
            heavy_chain = str(framework.get("heavy_chain") or "A")
            light_chain = str(framework.get("light_chain") or "")
            ratio_fn = None
            if protocol in {"antibody", "vhh"}:
                try:
                    from analysis.antibody_metric import get_interface_residues as _get_interface_residues

                    ratio_fn = _get_interface_residues
                except Exception:
                    ratio_fn = None

            import re

            def _infer_sample_id(stem: str) -> str | None:
                m = re.search(r"_antigen_antibody_sample_(\d+)$", stem)
                if m:
                    return m.group(1)
                if stem.startswith(f"{name}_"):
                    suf = stem[len(name) + 1 :]
                    if suf.isdigit():
                        return suf
                m = re.search(r"_(\d+)$", stem)
                if m:
                    return m.group(1)
                return None

            for fp in sorted(out_dir.glob("*.pdb")):
                sid = _infer_sample_id(fp.stem)
                if sid is None:
                    continue
                if ledger.has_done(str(sid)):
                    continue
                metrics_min: dict[str, Any] = {"pdb_path": str(fp)}
                if ratio_fn is not None:
                    try:
                        _, _, ratio = ratio_fn(str(fp), 10, heavy_chain, light_chain)
                        metrics_min["cdr_interface_ratio"] = float(ratio)
                    except Exception:
                        pass
                ledger.upsert(
                    str(sid),
                    status="done",
                    metrics=metrics_min,
                    outputs={"pdb_path": str(fp)},
                    worker_id=MetricsLedger.default_worker_id(),
                    attempt=1,
                    design_id=extract_design_id(fp.stem),
                    structure_id=structure_id_from_name(fp.stem),
                )

            rows = []
            for row in ledger.iter_rows(status="done"):
                merged = dict(row.metrics or {})
                # Keep backward-compat keys in the main CSV (downstream expects these columns).
                if isinstance(row.outputs, dict):
                    for k, v in row.outputs.items():
                        merged.setdefault(str(k), v)
                merged.setdefault("sample_id", row.item_id)
                rows.append(merged)
            if not rows:
                return
            # Deterministic ordering helps diff/debug.
            def _sort_key(r: dict) -> tuple[int, str]:
                sid = str(r.get("sample_id") or "")
                return (int(sid) if sid.isdigit() else 10**12, sid)

            rows = sorted(rows, key=_sort_key)
            try:
                import pandas as pd

                df = pd.DataFrame(rows)
                out_csv = out_dir / "sample_metrics.csv"
                tmp = out_csv.parent / f"{out_csv.name}.tmp"
                df.to_csv(tmp, index=False, float_format="%.4f")
                os.replace(tmp, out_csv)
            except Exception as exc:
                raise StepError(f"Failed to export sample_metrics.csv from ledger: {exc}") from exc
        finally:
            try:
                ledger.checkpoint_and_truncate_wal()
            finally:
                ledger.close()

    def _finalize_work_queue_outputs(self, ctx: StepContext, wq, *, items: list[Any], allow_failures: bool) -> None:
        # Strict finalize: produce derived outputs (sample_metrics.csv + manifest) before metadata.
        try:
            counts = wq.counts()
        except Exception:
            counts = None
        if counts and (counts.get("pending", 0) != 0 or counts.get("running", 0) != 0):
            return
        if not wq.acquire_leader():
            return
        try:
            # Export metrics first (canonical -> derived).
            self._export_sample_metrics_csv(ctx)
            if self.cfg.get("manifest"):
                self.write_manifest(ctx)
            if allow_failures:
                failed_ids: list[str] = []
                try:
                    for work_item, status in wq.iter_items():
                        if status in {"failed", "blocked"}:
                            failed_ids.append(str(work_item.id))
                except Exception:
                    failed_ids = []
                failed_path = self._failed_items_path(ctx)
                if failed_path is not None:
                    if failed_ids:
                        self._write_failed_items(ctx, failed_ids)
                    else:
                        try:
                            failed_path.unlink()
                        except FileNotFoundError:
                            pass
                        except Exception:
                            pass
            self._write_output_meta(ctx, items=items)
        finally:
            wq.release_leader()
