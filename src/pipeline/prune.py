from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from .direct_legacy import promote_tree
from .io import ensure_dir
from .output_policy import mode as output_mode
from .output_policy import optional_dir, should_keep, step_scratch_dir
from .work_queue import WorkQueue


_OPTIONAL_DIRS: dict[str, list[str]] = {
    "flowpacker1": ["after_pdbs", "flowpacker_outputs", ".tmp"],
    "flowpacker2": ["after_pdbs", "flowpacker_outputs", ".tmp"],
    "af3score1": ["af3score_outputs", "json", "af3_input_batch", "single_chain_cif", "pdbs", "af3score_subprocess_logs", ".tmp"],
    "af3score2": ["af3score_outputs", "json", "af3_input_batch", "single_chain_cif", "pdbs", "af3score_subprocess_logs", ".tmp"],
    "af3_refold": ["af3score_outputs", "json", "af3_input_batch", "single_chain_cif", "af3score_subprocess_logs", ".tmp"],
    "rosetta_interface": ["rosetta_jobs", ".tmp"],
    "relax": ["rosetta_jobs", ".tmp"],
    "seq1": [".tmp"],
    "seq2": [".tmp"],
}

_REQUIRED_OUTPUTS: dict[str, list[tuple[str, ...]]] = {
    "gen": [("sample_metrics.csv",)],
    "flowpacker1": [("packed_pdbs/*.pdb",)],
    "flowpacker2": [("packed_pdbs/*.pdb",)],
    "af3score1": [("metrics_items/*.csv",), ("cif/*.cif",), ("metrics.csv",), ("metrics_ppiflow.csv",), ("filtered_pdbs/*.pdb",)],
    "af3score2": [("metrics_items/*.csv",), ("cif/*.cif",), ("metrics.csv",), ("metrics_ppiflow.csv",), ("filtered_pdbs/*.pdb",)],
    "af3_refold": [("metrics_items/*.csv",), ("cif/*.cif",), ("pdbs/*.pdb",), ("metrics.csv",), ("metrics_ppiflow.csv",)],
    "rosetta_interface": [("residue_energy_items/*.csv", "residue_energy.csv")],
    "relax": [("*.pdb",)],
    "seq2": [("seqs/*.fa*",), ("pdbs/*.pdb",)],
}

_PARTIAL_OPTIONAL = ["wandb", "yaml", "config.yaml"]


def _resolve_output_dir(ctx, step_cfg: dict) -> Path | None:
    out_dir = step_cfg.get("output_dir")
    if not out_dir:
        return None
    p = Path(str(out_dir))
    if not p.is_absolute():
        p = ctx.out_dir / p
    return p


def _output_cfg(ctx) -> dict:
    raw = (ctx.input_data or {}).get("output")
    if isinstance(raw, dict):
        return raw
    return {}


def _prune_cfg(ctx) -> dict:
    cfg = _output_cfg(ctx)
    return {
        "dry_run": bool(cfg.get("prune_dry_run")),
        "keep_logs": bool(cfg.get("keep_logs", True)),
    }


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for fname in files:
            try:
                total += (Path(root) / fname).stat().st_size
            except Exception:
                pass
    return total


def _safe_rmtree(path: Path, *, dry_run: bool) -> int:
    if not path.exists():
        return 0
    size = _dir_size(path)
    if not dry_run:
        shutil.rmtree(path, ignore_errors=True)
    return size


def _safe_unlink(path: Path, *, dry_run: bool) -> int:
    if not path.exists():
        return 0
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    if not dry_run:
        try:
            path.unlink()
        except Exception:
            pass
    return size


def _move_file(src: Path, dst: Path, *, dry_run: bool) -> int:
    if not src.exists():
        return 0
    try:
        size = src.stat().st_size
    except Exception:
        size = 0
    if dry_run:
        return size
    ensure_dir(dst.parent)
    try:
        os.link(src, dst)
    except Exception:
        try:
            shutil.copy2(src, dst)
        except Exception:
            return size
    try:
        src.unlink()
    except Exception:
        pass
    return size


def _move_optional(src: Path, dst: Path, *, dry_run: bool) -> int:
    if not src.exists():
        return 0
    if src.is_file():
        return _move_file(src, dst, dry_run=dry_run)
    ensure_dir(dst.parent)
    size = _dir_size(src)
    if dry_run:
        return size
    try:
        promote_tree(src, dst, allow_reuse=True)
    except Exception:
        pass
    shutil.rmtree(src, ignore_errors=True)
    return size


def _log_prune(ctx, step_name: str, message: str, *, keep_logs: bool) -> None:
    if not keep_logs:
        return
    try:
        log_dir = ctx.out_dir / "logs"
        ensure_dir(log_dir)
        path = log_dir / "prune.log"
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} step={step_name} {message}\n"
        with path.open("a") as handle:
            handle.write(line)
    except Exception:
        pass


def _has_pattern(base: Path, pattern: str) -> bool:
    if any(ch in pattern for ch in "*?["):
        return any(base.glob(pattern))
    return (base / pattern).exists()


def _required_outputs_ready(ctx, out_dir: Path, step_name: str) -> bool:
    required = list(_REQUIRED_OUTPUTS.get(step_name) or [])
    if not required:
        return True
    for group in required:
        if not any(_has_pattern(out_dir, pattern) for pattern in group):
            return False
    return True


def _work_queue_idle(ctx, step_name: str) -> bool:
    cfg = ctx.work_queue or {}
    if not cfg.get("enabled"):
        return True
    step_dir = ctx.out_dir / ".work" / str(step_name)
    if not (step_dir / "queue.db").exists():
        return True
    try:
        wq = WorkQueue(ctx.out_dir, step_name, cfg)
        counts = wq.counts()
        return counts.get("pending", 0) == 0 and counts.get("running", 0) == 0
    except Exception:
        return False


def _handle_optional(
    ctx,
    path: Path,
    key: str,
    *,
    mode: str,
    opt_dest: Path,
    stats: dict,
    dry_run: bool,
) -> None:
    if not path.exists():
        return
    if mode == "minimal":
        if should_keep(ctx, key):
            moved = _move_optional(path, opt_dest, dry_run=dry_run)
            stats["moved_bytes"] += moved
            stats["moved"].append(str(path))
        else:
            if path.is_dir():
                stats["freed_bytes"] += _safe_rmtree(path, dry_run=dry_run)
            else:
                stats["freed_bytes"] += _safe_unlink(path, dry_run=dry_run)
            stats["removed"].append(str(path))
    else:
        moved = _move_optional(path, opt_dest, dry_run=dry_run)
        stats["moved_bytes"] += moved
        stats["moved"].append(str(path))


def _cleanup_partial_outputs(ctx, out_dir: Path, *, mode: str, opt_root: Path, stats: dict, dry_run: bool) -> None:
    for sid_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        sid = sid_dir.name
        for name in _PARTIAL_OPTIONAL:
            path = sid_dir / name
            _handle_optional(
                ctx,
                path,
                name,
                mode=mode,
                opt_dest=opt_root / sid / name,
                stats=stats,
                dry_run=dry_run,
            )

        input_dir = sid_dir / "input"
        if input_dir.exists():
            for child in sorted(input_dir.iterdir()):
                if child.is_file() and child.name.endswith("_input.csv"):
                    continue
                _handle_optional(
                    ctx,
                    child,
                    "input",
                    mode=mode,
                    opt_dest=opt_root / sid / "input" / child.name,
                    stats=stats,
                    dry_run=dry_run,
                )


def mark_rank_done(ctx, step_name: str) -> Path:
    step_dir = ctx.out_dir / ".work" / str(step_name)
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / f"step_done_rank{ctx.rank}"
    try:
        path.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except Exception:
        pass
    return path


def wait_for_all_ranks(ctx, step_name: str, world_size: int, timeout: float = 600.0) -> bool:
    step_dir = ctx.out_dir / ".work" / str(step_name)
    start = time.time()
    while (time.time() - start) < timeout:
        ok = True
        for rank in range(world_size):
            if not (step_dir / f"step_done_rank{rank}").exists():
                ok = False
                break
        if ok:
            return True
        time.sleep(2.0)
    return False


def step_cleanup(ctx, step_name: str, step_cfg: dict) -> dict:
    stats: dict[str, Any] = {
        "freed_bytes": 0,
        "moved_bytes": 0,
        "removed": [],
        "moved": [],
        "skipped": False,
        "reason": None,
        "dry_run": False,
    }
    out_dir = _resolve_output_dir(ctx, step_cfg)
    if out_dir is None:
        return stats

    mode = output_mode(ctx)
    prune_cfg = _prune_cfg(ctx)
    dry_run = bool(prune_cfg.get("dry_run"))
    keep_logs = bool(prune_cfg.get("keep_logs"))
    stats["dry_run"] = dry_run

    if not _work_queue_idle(ctx, step_name):
        stats["skipped"] = True
        stats["reason"] = "work_queue_active"
        _log_prune(ctx, step_name, "skip reason=work_queue_active", keep_logs=keep_logs)
        return stats

    if mode == "minimal" and not _required_outputs_ready(ctx, out_dir, step_name):
        stats["skipped"] = True
        stats["reason"] = "missing_required_outputs"
        _log_prune(ctx, step_name, "skip reason=missing_required_outputs", keep_logs=keep_logs)
        return stats

    opt_root = optional_dir(ctx) / out_dir.name

    if step_name == "partial":
        _cleanup_partial_outputs(ctx, out_dir, mode=mode, opt_root=opt_root, stats=stats, dry_run=dry_run)
    else:
        optional_names = _OPTIONAL_DIRS.get(step_name, [])
        for name in optional_names:
            path = out_dir / name
            _handle_optional(
                ctx,
                path,
                name,
                mode=mode,
                opt_dest=opt_root / name,
                stats=stats,
                dry_run=dry_run,
            )

    # Clean per-step scratch
    scratch_path = step_scratch_dir(ctx, step_name)
    if scratch_path.exists():
        stats["freed_bytes"] += _safe_rmtree(scratch_path, dry_run=dry_run)
        stats["removed"].append(str(scratch_path))

    # Optional log cleanup
    if not keep_logs:
        log_path = step_cfg.get("_log_file")
        if log_path:
            stats["freed_bytes"] += _safe_unlink(Path(str(log_path)), dry_run=dry_run)
            stats["removed"].append(str(log_path))

    if stats["removed"] or stats["moved"]:
        action = "dry_run" if dry_run else "applied"
        msg = f"{action} freed={stats['freed_bytes']}B moved={stats['moved_bytes']}B"
        _log_prune(ctx, step_name, msg, keep_logs=keep_logs)

    return stats
