from __future__ import annotations

from pathlib import Path
from typing import Any


def _as_input(data_or_ctx: Any) -> dict:
    if hasattr(data_or_ctx, "input_data"):
        return getattr(data_or_ctx, "input_data") or {}
    if isinstance(data_or_ctx, dict):
        return data_or_ctx
    return {}


def _as_out_dir(data_or_ctx: Any, out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    if hasattr(data_or_ctx, "out_dir"):
        return Path(getattr(data_or_ctx, "out_dir"))
    return Path(".")


def _output_cfg(data_or_ctx: Any) -> dict:
    input_data = _as_input(data_or_ctx)
    cfg = input_data.get("output")
    if isinstance(cfg, dict):
        return cfg
    return {}


def mode(data_or_ctx: Any, *, default: str = "minimal") -> str:
    cfg = _output_cfg(data_or_ctx)
    raw = str(cfg.get("mode") or default).strip().lower()
    if raw not in {"minimal", "full"}:
        return default
    return raw


def is_minimal(data_or_ctx: Any) -> bool:
    return mode(data_or_ctx) == "minimal"


def scratch_dir(data_or_ctx: Any, *, out_dir: str | Path | None = None) -> Path:
    cfg = _output_cfg(data_or_ctx)
    raw = cfg.get("scratch_dir")
    base = _as_out_dir(data_or_ctx, out_dir=out_dir).resolve()
    if raw:
        p = Path(str(raw))
        if not p.is_absolute():
            p = base / p
        return p.resolve()
    return (base / "scratch").resolve()


def optional_dir(data_or_ctx: Any, *, out_dir: str | Path | None = None) -> Path:
    base = _as_out_dir(data_or_ctx, out_dir=out_dir).resolve()
    return base / "output" / "_optional"


def should_keep(data_or_ctx: Any, key: str) -> bool:
    cfg = _output_cfg(data_or_ctx)
    keep = cfg.get("keep_optional") or []
    if not isinstance(keep, (list, tuple, set)):
        return False
    return str(key) in {str(k) for k in keep}


def step_scratch_dir(data_or_ctx: Any, step_name: str) -> Path:
    base = scratch_dir(data_or_ctx)
    return base / str(step_name)
