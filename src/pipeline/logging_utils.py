from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def _append_progress_log(line: str) -> None:
    path = os.environ.get("PPIFLOW_PROGRESS_LOG_PATH")
    if not path:
        return
    try:
        from fcntl import LOCK_EX, LOCK_UN, flock  # type: ignore
    except Exception:
        LOCK_EX = LOCK_UN = None  # type: ignore
        flock = None  # type: ignore
    try:
        with open(path, "a") as handle:
            if flock and LOCK_EX is not None:
                flock(handle, LOCK_EX)
            handle.write(line + "\n")
            handle.flush()
            if flock and LOCK_UN is not None:
                flock(handle, LOCK_UN)
    except Exception:
        pass


def run_command(
    cmd: Iterable[str],
    *,
    env: dict | None = None,
    cwd: str | Path | None = None,
    log_file: str | Path | None = None,
    verbose: bool = False,
    prefix: str | None = None,
) -> None:
    if prefix is None:
        prefix = os.environ.get("PPIFLOW_LOG_PREFIX")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if prefix:
            with log_path.open("a") as handle:
                proc = subprocess.Popen(
                    list(cmd),
                    cwd=str(cwd) if cwd else None,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    out_line = f"[{prefix}] {line}"
                    handle.write(out_line)
                    if verbose:
                        sys.stdout.write(out_line)
                ret = proc.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, list(cmd))
        elif verbose:
            with log_path.open("a") as handle:
                proc = subprocess.Popen(
                    list(cmd),
                    cwd=str(cwd) if cwd else None,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    handle.write(line)
                ret = proc.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, list(cmd))
        else:
            with log_path.open("a") as handle:
                subprocess.check_call(
                    list(cmd),
                    cwd=str(cwd) if cwd else None,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
    else:
        subprocess.check_call(list(cmd), cwd=str(cwd) if cwd else None, env=env)


def log_command_progress(
    step: str,
    idx: int,
    total: int,
    *,
    item: str | None = None,
    phase: str | None = None,
    status: str = "OK",
    elapsed: float | None = None,
    log_file: str | Path | None = None,
    extra: str | None = None,
) -> None:
    parts = [f"[{step}] {idx}/{total}", status]
    if elapsed is not None:
        parts.append(f"elapsed={elapsed:.2f}s")
    if item:
        parts.append(f"item={item}")
    if phase:
        parts.append(f"phase={phase}")
    if extra:
        parts.append(extra)
    if status != "OK" and log_file:
        parts.append(f"log={log_file}")
    line = " ".join(parts)
    print(line, file=sys.__stdout__, flush=True)
    _append_progress_log(line)
