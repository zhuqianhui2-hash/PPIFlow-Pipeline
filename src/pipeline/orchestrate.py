from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .configure import configure_pipeline
from .io import ensure_dir, read_json, read_yaml, repo_root, write_json
from .steps import STEP_REGISTRY
from .steps.base import StepContext
from .work_queue import WaitResult, wait_for_step, WorkQueue


class OrchestratorError(RuntimeError):
    pass


@dataclass
class FailurePolicy:
    mode: str
    max_failed: Optional[int] = None
    max_failed_ratio: Optional[float] = None


@dataclass
class PlanEntry:
    name: str
    steps: list[str]
    pool_size: int


def _load_steps(out_dir: Path) -> list[str]:
    steps_yaml = out_dir / "steps.yaml"
    if not steps_yaml.exists():
        raise OrchestratorError(f"steps.yaml not found at {steps_yaml}")
    steps_data = read_yaml(steps_yaml)
    steps_list = steps_data.get("steps") if isinstance(steps_data, dict) else None
    if not steps_list:
        raise OrchestratorError("steps.yaml is empty or invalid")
    steps: list[str] = []
    for entry in steps_list:
        name = entry.get("name") if isinstance(entry, dict) else None
        if name:
            steps.append(str(name))
    return steps


def _load_step_configs(out_dir: Path) -> dict[str, dict[str, Any]]:
    steps_yaml = out_dir / "steps.yaml"
    if not steps_yaml.exists():
        raise OrchestratorError(f"steps.yaml not found at {steps_yaml}")
    steps_data = read_yaml(steps_yaml)
    steps_list = steps_data.get("steps") if isinstance(steps_data, dict) else None
    if not steps_list:
        raise OrchestratorError("steps.yaml is empty or invalid")
    cfgs: dict[str, dict[str, Any]] = {}
    for entry in steps_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        cfg: dict[str, Any] = dict(entry)
        cfg_path = entry.get("config_file")
        if cfg_path:
            p = Path(str(cfg_path))
            if not p.is_absolute():
                p = out_dir / p
            if p.exists():
                try:
                    loaded = read_yaml(p)
                    if isinstance(loaded, dict):
                        cfg = loaded
                except Exception:
                    pass
            cfg["config_path"] = str(p)
        if "name" not in cfg:
            cfg["name"] = name
        cfgs[str(name)] = cfg
    return cfgs


def _build_ready_context(out_dir: Path, input_data: dict, work_queue_cfg: dict) -> StepContext:
    state_path = out_dir / "pipeline_state.json"
    state = {}
    if state_path.exists():
        try:
            state = read_json(state_path) or {}
        except Exception:
            state = {}
    run_id = int((state.get("runs") or [{"run_id": 0}])[0].get("run_id", 0) or 0)
    return StepContext(
        out_dir=out_dir,
        input_data=input_data,
        state=state,
        run_id=run_id,
        rank=0,
        world_size=1,
        local_rank=0,
        reuse=False,
        heartbeat=None,
        work_queue=work_queue_cfg,
    )


def _wait_for_ready_outputs(
    *,
    step_name: str,
    step_cfg: dict[str, Any],
    ctx: StepContext,
    timeout: Optional[float],
    poll_seconds: float = 2.0,
) -> bool:
    step_cls = STEP_REGISTRY.get(step_name)
    if not step_cls:
        return True
    step = step_cls(step_cfg)
    start = time.time()
    warned = False
    while True:
        try:
            if step.outputs_complete(ctx):
                return True
        except Exception:
            pass
        if timeout is not None and (time.time() - start) >= float(timeout):
            return False
        if not warned:
            print(
                f"[orchestrator] waiting for outputs of {step_name}",
                file=sys.__stdout__,
                flush=True,
            )
            warned = True
        time.sleep(max(float(poll_seconds), 0.5))


def _outputs_complete(
    *,
    step_name: str,
    step_cfg: dict[str, Any],
    ctx: StepContext,
) -> bool:
    step_cls = STEP_REGISTRY.get(step_name)
    if not step_cls:
        return True
    step = step_cls(step_cfg)
    try:
        return step.outputs_complete(ctx)
    except Exception:
        return False


def _normalize_failure_policy(raw: Dict[str, Any], override_mode: Optional[str]) -> FailurePolicy:
    mode = str(raw.get("mode") or "allow").lower()
    if override_mode:
        mode = str(override_mode).lower()
    if mode not in {"allow", "strict", "threshold"}:
        mode = "allow"
    max_failed = raw.get("max_failed")
    max_failed_ratio = raw.get("max_failed_ratio")
    if max_failed is not None:
        try:
            max_failed = int(max_failed)
        except Exception:
            max_failed = None
    if max_failed_ratio is not None:
        try:
            max_failed_ratio = float(max_failed_ratio)
        except Exception:
            max_failed_ratio = None
    return FailurePolicy(mode=mode, max_failed=max_failed, max_failed_ratio=max_failed_ratio)


def _policy_allows(policy: FailurePolicy, counts: Optional[Dict[str, int]]) -> bool:
    if policy.mode == "allow" or not counts:
        return True
    failed = int(counts.get("failed", 0) or 0)
    blocked = int(counts.get("blocked", 0) or 0)
    total = sum(int(v or 0) for v in counts.values())
    failures = failed + blocked
    if policy.mode == "strict":
        return failures == 0
    if policy.mode == "threshold":
        if policy.max_failed is not None and failures > policy.max_failed:
            return False
        if policy.max_failed_ratio is not None and total > 0:
            if (failures / float(total)) > policy.max_failed_ratio:
                return False
        return True
    return True


def _parse_devices(devices: Optional[str]) -> Optional[list[str]]:
    if devices is None:
        return None
    if isinstance(devices, str):
        raw = devices.strip()
        if raw.lower() == "all":
            detected = _detect_visible_devices()
            return detected or None
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        return tokens or None
    return None


def _detect_visible_devices() -> list[str]:
    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        return [d.strip() for d in env_devices.split(",") if d.strip()]
    try:
        import torch

        count = int(torch.cuda.device_count() or 0)
        if count > 0:
            return [str(i) for i in range(count)]
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        lines = [line for line in output.splitlines() if line.strip().startswith("GPU")]
        if lines:
            return [str(i) for i in range(len(lines))]
    except Exception:
        pass
    return []


def _parse_num_devices(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip()
    return value if value else None


def _resolve_device_list(cfg: Dict[str, Any], args_devices: Optional[str]) -> list[str]:
    cli_devices = _parse_devices(args_devices)
    if cli_devices:
        return cli_devices
    binding = cfg.get("gpu_binding") or {}
    cfg_devices = binding.get("devices")
    if isinstance(cfg_devices, str) and cfg_devices.strip().lower() == "all":
        detected = _detect_visible_devices()
        return detected or []
    if isinstance(cfg_devices, list) and cfg_devices:
        return [str(d) for d in cfg_devices]
    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        return [d.strip() for d in env_devices.split(",") if d.strip()]
    return []


def _next_attempt(run_dir: Path) -> int:
    if not run_dir.exists():
        return 1
    existing = list(run_dir.glob("attempt_*.json"))
    if not existing:
        return 1
    max_idx = 0
    for path in existing:
        stem = path.stem
        try:
            idx = int(stem.split("_")[-1])
            max_idx = max(max_idx, idx)
        except Exception:
            continue
    return max_idx + 1


def _plan_from_config(available: list[str], cfg: Dict[str, Any]) -> list[PlanEntry]:
    plan: list[PlanEntry] = []
    steps_cfg = cfg.get("steps")
    if isinstance(steps_cfg, list) and steps_cfg:
        for entry in steps_cfg:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            if name not in available:
                raise OrchestratorError(f"Unknown step in orchestrator config: {name}")
            pool_size = int(entry.get("pool_size") or 1)
            plan.append(PlanEntry(name=name, steps=[name], pool_size=pool_size))
    return plan


def _build_plan(
    available: list[str],
    cfg: Dict[str, Any],
    *,
    single_step: Optional[str],
    pool_override: Optional[int],
) -> list[PlanEntry]:
    if single_step:
        if single_step not in available:
            raise OrchestratorError(f"Unknown step: {single_step}")
        pool_size = int(pool_override or 1)
        return [PlanEntry(name=single_step, steps=[single_step], pool_size=pool_size)]

    plan = _plan_from_config(available, cfg)
    if not plan:
        plan = [PlanEntry(name=s, steps=[s], pool_size=1) for s in available]
    if pool_override is not None:
        for entry in plan:
            entry.pool_size = int(pool_override)
    return plan


def _build_worker_cmd(
    out_dir: Path,
    steps: list[str],
    *,
    continue_on_error: bool,
    wait_timeout: Optional[int],
    work_queue_rebuild: bool = False,
) -> list[str]:
    root = repo_root(out_dir)
    ppiflow = root / "ppiflow.py"
    cmd = [sys.executable, str(ppiflow), "execute", "--output", str(out_dir), "--steps", ",".join(steps), "--work-queue"]
    if continue_on_error:
        cmd.append("--continue-on-error")
    if wait_timeout is not None:
        cmd.extend(["--work-queue-wait-timeout", str(wait_timeout)])
    if work_queue_rebuild:
        cmd.append("--work-queue-rebuild")
    return cmd


def _spawn_workers(
    *,
    out_dir: Path,
    steps: list[str],
    pool_size: int,
    bind: bool,
    devices: list[str],
    continue_on_error: bool,
    wait_timeout: Optional[int],
    run_dir: Path,
    log_ctx: Dict[str, Any],
    progress_log_path: Path | None = None,
    work_queue_rebuild: bool = False,
    retry_failed_override: Optional[bool] = None,
) -> tuple[list[subprocess.Popen], list[dict]]:
    procs: list[subprocess.Popen] = []
    workers: list[dict] = []
    if pool_size < 1:
        pool_size = 1
    device_count = max(len(devices), 0)

    for rank in range(pool_size):
        env = os.environ.copy()
        device = None
        if bind:
            env["WORLD_SIZE"] = str(pool_size)
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank % max(device_count, 1))
            if devices:
                device = devices[rank % len(devices)]
                env["CUDA_VISIBLE_DEVICES"] = str(device)
        else:
            device = env.get("CUDA_VISIBLE_DEVICES")
        if retry_failed_override is not None:
            env["PPIFLOW_WORK_QUEUE_RETRY_FAILED"] = "1" if retry_failed_override else "0"
        if progress_log_path is not None:
            env["PPIFLOW_PROGRESS_LOG_PATH"] = str(progress_log_path)
        cmd = _build_worker_cmd(
            out_dir,
            steps,
            continue_on_error=continue_on_error,
            wait_timeout=wait_timeout,
            work_queue_rebuild=work_queue_rebuild,
        )
        stdout_path = run_dir / f"stdout_rank{rank}.log"
        stderr_path = run_dir / f"stderr_rank{rank}.log"
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(proc)
        workers.append(
            {
                "rank": rank,
                "pid": proc.pid,
                "device": device,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "cmd": cmd,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
        )
        _start_log_threads(
            proc,
            rank=rank,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            log_ctx=log_ctx,
        )
    return procs, workers


def _terminate_workers(procs: Iterable[subprocess.Popen], *, grace: float = 10.0) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    start = time.time()
    for proc in procs:
        try:
            proc.wait(timeout=max(grace - (time.time() - start), 0.1))
        except Exception:
            proc.kill()


def _clear_failure_markers(step_dir: Path) -> None:
    try:
        wq = WorkQueue.from_step_dir(step_dir)
        wq.reset_items_for_retry()
        wq.reset_leader_for_retry()
    except Exception:
        pass


def _reset_attempt_logs(run_dir: Path) -> None:
    # Intentionally do not truncate logs; append across attempts.
    return


def _open_log_context(
    run_dir: Path,
    *,
    run_log_path: Path | None = None,
    step_name: str | None = None,
    attempt: int | None = None,
) -> Dict[str, Any]:
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    ctx = {
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_handle": stdout_path.open("a"),
        "stderr_handle": stderr_path.open("a"),
        "lock": threading.Lock(),
        "threads": [],
        "run_log_path": str(run_log_path) if run_log_path else None,
        "run_log_handle": run_log_path.open("a") if run_log_path else None,
        "step_name": step_name,
    }
    if attempt is not None:
        header = f"===== {step_name or run_dir.name} attempt {attempt} =====\n"
        try:
            ctx["stdout_handle"].write(header)
            ctx["stdout_handle"].flush()
        except Exception:
            pass
        try:
            if ctx["run_log_handle"] is not None:
                ctx["run_log_handle"].write(header)
                ctx["run_log_handle"].flush()
        except Exception:
            pass
    return ctx


def _close_log_context(log_ctx: Dict[str, Any]) -> None:
    for thread in log_ctx.get("threads") or []:
        thread.join(timeout=2.0)
    for key in ("stdout_handle", "stderr_handle", "run_log_handle"):
        handle = log_ctx.get(key)
        try:
            if handle:
                handle.close()
        except Exception:
            pass


def _start_log_threads(
    proc: subprocess.Popen,
    *,
    rank: int,
    stdout_path: Path,
    stderr_path: Path,
    log_ctx: Dict[str, Any],
) -> None:
    stdout_handle = log_ctx["stdout_handle"]
    stderr_handle = log_ctx["stderr_handle"]
    lock = log_ctx["lock"]
    run_log_handle = log_ctx.get("run_log_handle")
    step_name = log_ctx.get("step_name")

    def _fan_in(pipe, per_rank_path: Path, agg_handle, stream_label: str) -> None:
        if pipe is None:
            return
        try:
            wrapper = io.TextIOWrapper(pipe, encoding="utf-8", errors="replace")
        except Exception:
            wrapper = pipe
        try:
            with per_rank_path.open("a") as per_rank:
                for line in wrapper:
                    if not line.endswith("\n"):
                        line = line + "\n"
                    per_rank.write(line)
                    per_rank.flush()
                    with lock:
                        agg_handle.write(f"[rank={rank}] {line}")
                        agg_handle.flush()
                        if run_log_handle is not None:
                            prefix = f"[step={step_name or 'unknown'} rank={rank}"
                            if stream_label:
                                prefix = f"{prefix} {stream_label}] "
                            else:
                                prefix = f"{prefix}] "
                            run_log_handle.write(prefix + line)
                            run_log_handle.flush()
        finally:
            try:
                wrapper.close()
            except Exception:
                pass

    if proc.stdout is not None:
        t_out = threading.Thread(
            target=_fan_in,
            args=(proc.stdout, stdout_path, stdout_handle, ""),
            daemon=True,
        )
        log_ctx["threads"].append(t_out)
        t_out.start()
    if proc.stderr is not None:
        t_err = threading.Thread(
            target=_fan_in,
            args=(proc.stderr, stderr_path, stderr_handle, "stderr"),
            daemon=True,
        )
        log_ctx["threads"].append(t_err)
        t_err.start()


def _wait_for_step_with_process_check(
    *,
    step_dir: Path,
    step_name: str,
    step_cfg: dict[str, Any],
    ctx: StepContext,
    procs: list[subprocess.Popen],
    timeout: Optional[float],
    poll_seconds: float = 2.0,
) -> WaitResult:
    start = time.time()
    while True:
        result = wait_for_step(step_dir, timeout=0)
        if result.status in {"completed", "failed"}:
            if result.status == "failed":
                if _outputs_complete(step_name=step_name, step_cfg=step_cfg, ctx=ctx):
                    return WaitResult(status="completed", reason=None, complete=True)
            return result
        if _outputs_complete(step_name=step_name, step_cfg=step_cfg, ctx=ctx):
            return WaitResult(status="completed", reason=None, complete=True)
        if all(proc.poll() is not None for proc in procs):
            if _outputs_complete(step_name=step_name, step_cfg=step_cfg, ctx=ctx):
                return WaitResult(status="completed", reason=None, complete=True)
            return WaitResult(status="failed", reason="all workers exited")
        if timeout is not None and (time.time() - start) >= float(timeout):
            return WaitResult(status="timeout", reason="timeout")
        time.sleep(max(float(poll_seconds), 0.5))


def orchestrate_pipeline(args) -> None:
    out_dir = Path(args.output).resolve()
    ensure_dir(out_dir)
    log_dir = out_dir / "logs"
    ensure_dir(log_dir)

    steps_yaml = out_dir / "steps.yaml"
    input_json = out_dir / "pipeline_input.json"

    if not steps_yaml.exists() or not input_json.exists():
        if getattr(args, "configure", False):
            if not getattr(args, "input", None):
                raise OrchestratorError("--configure requires --input")
            configure_pipeline(args)
        else:
            missing = "steps.yaml" if not steps_yaml.exists() else "pipeline_input.json"
            raise OrchestratorError(f"{missing} missing; run configure first or pass --configure")

    input_data = read_json(input_json)
    orch_cfg = dict(input_data.get("orchestrator") or {})
    if orch_cfg.get("groups"):
        raise OrchestratorError("orchestrator groups are not supported; use per-step pools only")
    work_queue_cfg = dict(input_data.get("work_queue") or {})
    work_queue_cfg.setdefault("enabled", True)
    work_queue_cfg.setdefault("lease_seconds", 1800)
    work_queue_cfg.setdefault("max_attempts", 1)
    work_queue_cfg.setdefault("retry_failed", False)
    work_queue_cfg.setdefault("batch_size", 1)
    work_queue_cfg.setdefault("leader_timeout", 600)
    work_queue_cfg.setdefault("wait_timeout", None)
    work_queue_cfg.setdefault("allow_reuse", True)
    work_queue_cfg.setdefault("rebuild_from_outputs", False)
    work_queue_cfg.setdefault("explicit_reuse", False)
    if getattr(args, "work_queue_reuse", False):
        work_queue_cfg["allow_reuse"] = True
        work_queue_cfg["explicit_reuse"] = True
    if getattr(args, "work_queue_strict", False):
        work_queue_cfg["allow_reuse"] = False
    if getattr(args, "work_queue_rebuild", False):
        work_queue_cfg["rebuild_from_outputs"] = True
    retry_failed_configured = bool(work_queue_cfg.get("retry_failed"))
    explicit_retry = bool(getattr(args, "retry_failed", False))
    if retry_failed_configured and not explicit_retry:
        print(
            "[orchestrator] note: disabling item-level retry_failed for worker runs",
            file=sys.__stdout__,
            flush=True,
        )

    failure_policy = _normalize_failure_policy(
        orch_cfg.get("failure_policy") or {},
        getattr(args, "failure_policy", None),
    )

    continue_on_error = failure_policy.mode in {"allow", "threshold"}
    if continue_on_error:
        # Keep ready_ctx consistent with worker behavior when we spawn with --continue-on-error.
        options = input_data.setdefault("options", {})
        options.setdefault("continue_on_item_error", True)

    if getattr(args, "max_retries", None) is not None:
        orch_cfg.setdefault("retries", {})["max_step_attempts"] = int(args.max_retries)

    available_steps = _load_steps(out_dir)
    step_cfgs = _load_step_configs(out_dir)
    ready_ctx = _build_ready_context(out_dir, input_data, work_queue_cfg)

    step_arg = getattr(args, "steps", None)
    if step_arg and step_arg not in {"", "all"}:
        if "," in step_arg:
            raise OrchestratorError("--steps accepts a single step only")
        single_step = step_arg.strip()
    else:
        single_step = None

    num_devices_raw = _parse_num_devices(getattr(args, "num_devices", None))
    devices_arg = getattr(args, "devices", None)
    pool_override = getattr(args, "pool_size", None)

    if num_devices_raw is not None and getattr(args, "no_bind", False):
        raise OrchestratorError("--num-devices requires GPU binding (remove --no-bind)")

    if num_devices_raw is not None:
        explicit_devices = _parse_devices(devices_arg)
        if explicit_devices is not None and len(explicit_devices) == 0:
            explicit_devices = None

        if num_devices_raw.lower() == "all":
            devices = explicit_devices or _detect_visible_devices()
            if not devices:
                raise OrchestratorError("No GPUs detected for --num-devices all")
            if explicit_devices is not None and len(explicit_devices) != len(devices):
                raise OrchestratorError("--devices and --num-devices all conflict")
        else:
            try:
                num_devices = int(num_devices_raw)
            except Exception as exc:
                raise OrchestratorError(f"Invalid --num-devices value: {num_devices_raw}") from exc
            if num_devices < 1:
                raise OrchestratorError("--num-devices must be >= 1")
            if explicit_devices is not None:
                if len(explicit_devices) != num_devices:
                    raise OrchestratorError("--devices count must match --num-devices")
                devices = explicit_devices
            else:
                visible = _detect_visible_devices()
                if len(visible) < num_devices:
                    raise OrchestratorError(
                        f"--num-devices {num_devices} requested but only {len(visible)} visible GPUs"
                    )
                devices = visible[:num_devices]

        if pool_override is not None and int(pool_override) != len(devices):
            raise OrchestratorError("--pool-size must match --num-devices")
        pool_override = len(devices)
    else:
        devices = _resolve_device_list(orch_cfg, devices_arg)

    plan = _build_plan(
        available_steps,
        orch_cfg,
        single_step=single_step,
        pool_override=pool_override,
    )

    orch_dir = out_dir / ".orchestrator"
    ensure_dir(orch_dir)
    plan_payload = {
        "output": str(out_dir),
        "plan": [
            {
                "name": entry.name,
                "steps": entry.steps,
                "pool_size": entry.pool_size,
            }
            for entry in plan
        ],
        "failure_policy": {
            "mode": failure_policy.mode,
            "max_failed": failure_policy.max_failed,
            "max_failed_ratio": failure_policy.max_failed_ratio,
        },
    }
    write_json(orch_dir / "plan.json", plan_payload, indent=2)

    retries = orch_cfg.get("retries") or {}
    max_attempts = int(retries.get("max_step_attempts") or 1)
    timeouts = orch_cfg.get("timeouts") or {}
    step_wait_timeout = timeouts.get("step_wait_timeout")
    if step_wait_timeout is not None:
        try:
            step_wait_timeout = int(step_wait_timeout)
        except Exception:
            step_wait_timeout = None

    bind = bool((orch_cfg.get("gpu_binding") or {}).get("enabled", True))
    if getattr(args, "no_bind", False):
        bind = False

    if bind:
        if not devices:
            pool_sizes = {entry.pool_size for entry in plan}
            if len(pool_sizes) != 1:
                raise OrchestratorError(
                    "GPU binding requires a single pool size (use --num-devices or --pool-size)"
                )
            pool_size = next(iter(pool_sizes))
            visible = _detect_visible_devices()
            if len(visible) < pool_size:
                raise OrchestratorError(
                    f"Pool size {pool_size} requires {pool_size} GPUs but only {len(visible)} visible"
                )
            devices = visible[:pool_size]
        device_count = len(devices)
        for entry in plan:
            if entry.pool_size != device_count:
                raise OrchestratorError(
                    f"Pool size {entry.pool_size} must match device count {device_count} "
                    "(use --num-devices or --pool-size to align)"
                )

    summary: dict[str, Any] = {"status": "running", "steps": []}

    rebuild_requested = bool(getattr(args, "work_queue_rebuild", False))

    for entry in plan:
        entry_complete = True
        for step_name in entry.steps:
            step_cfg = step_cfgs.get(step_name) or {"name": step_name}
            if not _outputs_complete(step_name=step_name, step_cfg=step_cfg, ctx=ready_ctx):
                entry_complete = False
                break
        if entry_complete:
            summary["steps"].append({"name": entry.name, "status": "ok", "attempt": 0, "skipped": True})
            continue

        attempt = 0
        step_success = False
        while attempt < max_attempts and not step_success:
            attempt += 1
            if attempt > 1:
                for step_name in entry.steps:
                    _clear_failure_markers(out_dir / ".work" / step_name)
            run_dir = orch_dir / "runs" / entry.name
            ensure_dir(run_dir)
            attempt_idx = _next_attempt(run_dir)
            attempt_path = run_dir / f"attempt_{attempt_idx:04d}.json"
            _reset_attempt_logs(run_dir)
            run_log_path = log_dir / "run.log"
            progress_log_path = log_dir / f"{entry.name}.progress.log"
            log_ctx = _open_log_context(
                run_dir,
                run_log_path=run_log_path,
                step_name=entry.name,
                attempt=attempt,
            )

            procs, workers = _spawn_workers(
                out_dir=out_dir,
                steps=entry.steps,
                pool_size=entry.pool_size,
                bind=bind,
                devices=devices,
                continue_on_error=continue_on_error,
                wait_timeout=step_wait_timeout,
                run_dir=run_dir,
                log_ctx=log_ctx,
                progress_log_path=progress_log_path,
                work_queue_rebuild=(rebuild_requested and attempt == 1),
                retry_failed_override=True if explicit_retry else False,
            )
            write_json(run_dir / "workers.json", {"workers": workers}, indent=2)

            start_ts = time.time()
            step_results: list[dict] = []
            failed_reason = None
            step_ok = True
            for step_name in entry.steps:
                step_dir = out_dir / ".work" / step_name
                result = _wait_for_step_with_process_check(
                    step_dir=step_dir,
                    step_name=step_name,
                    step_cfg=step_cfgs.get(step_name) or {"name": step_name},
                    ctx=ready_ctx,
                    procs=procs,
                    timeout=step_wait_timeout,
                )
                step_info = {
                    "step": step_name,
                    "status": result.status,
                    "reason": result.reason,
                    "counts": result.counts,
                }
                step_results.append(step_info)
                if result.status == "failed":
                    step_ok = False
                    failed_reason = result.reason or "failed"
                    break
                if result.status == "timeout":
                    step_ok = False
                    failed_reason = "timeout"
                    break
                if result.status == "completed":
                    if not _policy_allows(failure_policy, result.counts):
                        step_ok = False
                        failed_reason = "failure_policy"
                        break
                    step_cfg = step_cfgs.get(step_name) or {"name": step_name}
                    if not _wait_for_ready_outputs(
                        step_name=step_name,
                        step_cfg=step_cfg,
                        ctx=ready_ctx,
                        timeout=step_wait_timeout,
                    ):
                        step_ok = False
                        failed_reason = "outputs_not_ready"
                        step_info["status"] = "failed"
                        step_info["reason"] = failed_reason
                        break
            if not step_ok:
                _terminate_workers(procs)
            else:
                for proc in procs:
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        _terminate_workers([proc])
            _close_log_context(log_ctx)

            for idx, proc in enumerate(procs):
                workers[idx]["exit_code"] = proc.poll()
            write_json(run_dir / "workers.json", {"workers": workers}, indent=2)

            elapsed = time.time() - start_ts
            attempt_payload = {
                "entry": {
                    "name": entry.name,
                    "steps": entry.steps,
                    "pool_size": entry.pool_size,
                },
                "attempt": attempt,
                "started_at": start_ts,
                "elapsed_seconds": elapsed,
                "status": "ok" if step_ok else "failed",
                "reason": failed_reason,
                "steps": step_results,
                "workers": workers,
                "logs": {
                    "stdout": log_ctx["stdout_path"],
                    "stderr": log_ctx["stderr_path"],
                    "run": str(run_log_path),
                    "progress": str(progress_log_path),
                },
            }
            write_json(attempt_path, attempt_payload, indent=2)

            if step_ok:
                summary["steps"].append({"name": entry.name, "status": "ok", "attempt": attempt})
                step_success = True
            else:
                summary["steps"].append({"name": entry.name, "status": "failed", "attempt": attempt})
                if attempt >= max_attempts:
                    summary["status"] = "failed"
                    summary["reason"] = failed_reason
                    write_json(orch_dir / "orchestrator.json", summary, indent=2)
                    raise OrchestratorError(f"Step {entry.name} failed: {failed_reason}")

        if not step_success:
            break

    if summary.get("status") != "failed":
        summary["status"] = "completed"
    write_json(orch_dir / "orchestrator.json", summary, indent=2)
