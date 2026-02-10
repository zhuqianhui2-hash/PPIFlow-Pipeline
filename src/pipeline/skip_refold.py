from __future__ import annotations

from typing import Any, Iterable

from .steps import STEP_ORDER


SKIP_REFOLD_STEPS = ("af3_refold", "dockq")


def resolve_steps_arg(steps_arg: str | None, *, available_steps: Iterable[str] = STEP_ORDER) -> list[str]:
    """
    Resolve a --steps argument into a canonical, de-duplicated, ordered step list.

    Semantics match execute/orchestrate:
    - None/"all"/empty -> all available steps
    - comma-list -> filter-only in canonical order
    """
    available = list(available_steps)
    if steps_arg is None:
        return list(available)
    raw = str(steps_arg).strip()
    if not raw or raw.lower() == "all":
        return list(available)
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    requested = set(tokens)
    unknown = sorted(requested - set(available))
    if unknown:
        bad = ", ".join(unknown)
        valid = ", ".join(available)
        raise ValueError(f"Unknown step(s) in --steps: {bad}\nValid steps: {valid}")
    # Preserve canonical ordering from STEP_ORDER / available_steps.
    return [s for s in available if s in requested]


def remove_skip_refold_steps(steps: list[str]) -> list[str]:
    banned = set(SKIP_REFOLD_STEPS)
    return [s for s in steps if s not in banned]


def steps_conflict_with_skip_refold(steps_arg: str | None) -> list[str]:
    """
    Return the subset of explicitly requested steps that are incompatible with --skip-refold.
    """
    if steps_arg is None:
        return []
    raw = str(steps_arg).strip()
    if not raw or raw.lower() == "all":
        return []
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    banned = set(SKIP_REFOLD_STEPS)
    return sorted([t for t in tokens if t in banned])


def apply_skip_refold_ranking_policy(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Mutate input_data in-place to force ranking to use AF3Score R2 metrics/structures and
    disable DockQ filtering.
    """
    pipeline_options = input_data.get("pipeline_options")
    if not isinstance(pipeline_options, dict):
        pipeline_options = {}
    pipeline_options["skip_refold"] = True
    input_data["pipeline_options"] = pipeline_options

    ranking = input_data.get("ranking")
    if not isinstance(ranking, dict):
        ranking = {}
    ranking["metrics_source"] = "af3score2"
    ranking["structure_source"] = "af3score2"
    input_data["ranking"] = ranking

    # Explicitly disable DockQ filtering in skip mode.
    filters = input_data.get("filters")
    if not isinstance(filters, dict):
        filters = {}
    dockq = filters.get("dockq")
    if not isinstance(dockq, dict):
        dockq = {}
    dockq["min"] = None
    filters["dockq"] = dockq
    input_data["filters"] = filters

    return input_data

