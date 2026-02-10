from __future__ import annotations

import argparse
import os
from pathlib import Path

from .configure import configure_pipeline
from .execute import execute_pipeline
from .orchestrate import orchestrate_pipeline
from .skip_refold import remove_skip_refold_steps, steps_conflict_with_skip_refold
from .steps import STEP_ORDER
from .wizard import run_wizard


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=str, choices=["binder", "antibody", "vhh"], help="Protocol")
    parser.add_argument("--preset", type=str, choices=["fast", "full", "custom"], default="full")
    parser.add_argument("--input", type=str, help="Path to design.yaml")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["minimal", "full"],
        default=None,
        help="Output mode (minimal or full). Default: minimal.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Step names (all or comma-separated step list)",
    )
    parser.add_argument(
        "--skip-refold",
        "--skip-af3-refold",
        dest="skip_refold",
        action="store_true",
        help="Skip AF3 refold + DockQ and force ranking to use AF3Score R2 metrics/structures.",
    )
    parser.add_argument("--reuse", action="store_true", help="Reuse existing outputs")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next step if a step fails",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream step subprocess output to console (also logs to file)",
    )
    parser.add_argument("--num-devices", type=str, default=None, help="Number of GPUs/workers (e.g. 4 or 'all')")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated GPU list or 'all'")


def _add_work_queue_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-queue", action="store_true", help="Enable work queue")
    parser.add_argument("--work-queue-lease-seconds", type=int, default=None)
    parser.add_argument("--work-queue-max-attempts", type=int, default=None)
    parser.add_argument("--work-queue-batch-size", type=int, default=None)
    parser.add_argument("--work-queue-leader-timeout", type=int, default=None)
    parser.add_argument("--work-queue-wait-timeout", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--work-queue-reuse", action="store_true")
    parser.add_argument("--work-queue-strict", action="store_true", help="Disable reuse; enforce strict checks")
    parser.add_argument("--work-queue-rebuild", action="store_true", help="Drop/rebuild queue.db from outputs")


def _add_orchestrator_pool_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-rosetta-workers",
        type=int,
        default=None,
        help="Orchestrator: pool size override for Rosetta (CPU) items steps (advanced)",
    )
    parser.add_argument(
        "--num-cpu-workers",
        type=int,
        default=None,
        help="Alias for --num-rosetta-workers",
    )


def _add_run_lock_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-run-lock", action="store_true", help="Disable output run lock (debug only)")
    parser.add_argument(
        "--run-lock-stale-seconds",
        type=int,
        default=None,
        help="Consider an existing lock stale after this many seconds (default: derived from heartbeat interval)",
    )
    parser.add_argument("--steal-lock", action="store_true", help="Take over a lock even if it looks active")


def _add_cli_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", type=str, help="Design name")
    parser.add_argument("--target_pdb", type=str, help="Target PDB path")
    parser.add_argument("--target_chains", type=str, help="Target chain(s) (comma-separated)")
    parser.add_argument("--hotspots", type=str, help="Hotspots (comma-separated)")
    parser.add_argument("--binder_length", type=str, help="Binder length (e.g. 75-90)")
    parser.add_argument("--samples_per_target", type=int, default=100)
    parser.add_argument("--partial_samples_per_target", type=int, help="Partial flow: samples per target (default 8)")
    parser.add_argument("--framework_pdb", type=str, help="Framework PDB")
    parser.add_argument("--heavy_chain", type=str, help="Heavy chain id")
    parser.add_argument("--light_chain", type=str, help="Light chain id")
    parser.add_argument("--cdr_length", type=str, help="CDR length spec")

    # Sequence design knobs (paper-aligned defaults in config)
    parser.add_argument("--seq1_num_per_backbone", type=int, help="Seq1: sequences per backbone")
    parser.add_argument("--seq1_temp", type=float, help="Seq1: sampling temperature")
    parser.add_argument("--seq1_bias_large_residues", action="store_true", help="Seq1: bias bulky residues")
    parser.add_argument("--seq1_bias_num", type=int, help="Seq1: number of biased sequences")
    parser.add_argument("--seq1_bias_residues", type=str, help="Seq1: bias residues (comma-separated)")
    parser.add_argument("--seq1_bias_weight", type=float, help="Seq1: bias weight for residues")
    parser.add_argument("--seq2_num_per_backbone", type=int, help="Seq2: sequences per backbone")
    parser.add_argument("--seq2_temp", type=float, help="Seq2: sampling temperature")
    parser.add_argument("--seq2_use_soluble_ckpt", action="store_true", help="Seq2: use soluble checkpoint")

    # Antibody/VHH aliases
    parser.add_argument("--vhh_backbones", type=int, help="Alias for samples_per_target (antibody/vhh)")
    parser.add_argument("--vhh_cdr1_num", type=int, help="Alias for seq1_num_per_backbone (antibody/vhh)")
    parser.add_argument("--vhh_partial_num", type=int, help="Alias for partial_samples_per_target (antibody/vhh)")
    parser.add_argument("--vhh_cdr2_num", type=int, help="Alias for seq2_num_per_backbone (antibody/vhh)")

    # Filtering / ranking knobs
    parser.add_argument("--af3score1_iptm_min", type=float, help="AF3Score R1 ipTM threshold")
    parser.add_argument("--af3score1_ptm_min", type=float, help="AF3Score R1 pTM threshold")
    parser.add_argument("--af3score1_top_k", type=int, help="AF3Score R1: keep top K candidates (global)")
    parser.add_argument("--af3score2_iptm_min", type=float, help="AF3Score R2 ipTM threshold")
    parser.add_argument("--af3score2_ptm_min", type=float, help="AF3Score R2 pTM threshold")
    parser.add_argument("--af3score2_top_k", type=int, help="AF3Score R2: keep top K candidates (global)")
    parser.add_argument("--af3refold_iptm_min", type=float, help="AF3 refold ipTM threshold")
    parser.add_argument("--af3refold_ptm_min", type=float, help="AF3 refold pTM threshold")
    parser.add_argument("--af3refold_dockq_min", type=float, help="AF3 refold DockQ threshold")
    parser.add_argument("--af3refold_num_samples", type=int, help="AF3 refold: num_samples per seed")
    parser.add_argument("--af3refold_model_seeds", type=str, help="AF3 refold: model seeds (e.g. 0-19)")
    parser.add_argument(
        "--af3refold_no_templates",
        action="store_true",
        default=None,
        help="AF3 refold: disable templates",
    )
    parser.add_argument(
        "--af3-num-workers",
        type=int,
        default=None,
        help="AF3Score: num_workers (advanced; defaults to run_af3score.py default)",
    )
    parser.add_argument("--interface_energy_min", type=float, help="Rosetta interface energy cutoff (REU)")
    parser.add_argument("--interface_distance", type=float, help="Rosetta interface distance cutoff (A)")
    parser.add_argument("--relax_max_iter", type=int, help="Rosetta relax max iterations (legacy)")
    parser.add_argument("--relax_fixbb", action="store_true", default=None, help="Rosetta relax: fix backbone (legacy)")
    parser.add_argument(
        "--fixed_chains",
        type=str,
        help="Chains to fix backbone (underscore or comma separated, e.g. A_B or A,B)",
    )
    parser.add_argument("--dockq_min", type=float, help="DockQ minimum threshold")
    parser.add_argument("--partial_start_t", type=float, help="Partial flow start_t")
    parser.add_argument("--rank_top_k", type=int, help="Top K to keep in ranking")


    # Tool paths (optional)
    parser.add_argument("--ppiflow_ckpt", type=str, help="PPIFlow checkpoint path")
    parser.add_argument("--abmpnn_ckpt", type=str, help="AbMPNN checkpoint path")
    parser.add_argument("--mpnn_ckpt", type=str, help="ProteinMPNN checkpoint path")
    parser.add_argument("--mpnn_ckpt_soluble", type=str, help="ProteinMPNN soluble checkpoint path")
    parser.add_argument("--af3score_repo", type=str, help="AF3Score repo path")
    parser.add_argument("--rosetta_bin", type=str, help="Rosetta scripts binary path (rosetta_scripts)")
    parser.add_argument("--rosetta_db", type=str, help="Rosetta database path (optional)")
    parser.add_argument("--flowpacker_repo", type=str, help="FlowPacker repo path")
    parser.add_argument("--af3_weights", type=str, help="AF3 weights path")
    parser.add_argument("--mpnn_repo", type=str, help="ProteinMPNN repo path")
    parser.add_argument("--abmpnn_repo", type=str, help="AbMPNN repo path")
    parser.add_argument("--mpnn_run", type=str, help="Path to protein_mpnn_run.py")
    parser.add_argument("--abmpnn_run", type=str, help="Path to protein_mpnn_run.py for AbMPNN")
    parser.add_argument("--dockq_bin", type=str, help="Path to DockQ binary")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("ppiflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pipeline = sub.add_parser("pipeline", help="Configure then execute")
    _add_common_args(p_pipeline)
    _add_orchestrator_pool_args(p_pipeline)
    p_pipeline.add_argument(
        "--single-process",
        action="store_true",
        help="Force legacy single-process execution (debug/regression only; disables orchestrator pools)",
    )
    p_pipeline.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip configure if config files already exist (auto-configure if missing)",
    )
    p_pipeline.add_argument(
        "--force-config",
        action="store_true",
        help="Always regenerate configs before running",
    )
    _add_work_queue_args(p_pipeline)
    _add_run_lock_args(p_pipeline)
    _add_cli_input_args(p_pipeline)

    p_configure = sub.add_parser("configure", help="Write step configs and steps.yaml")
    _add_common_args(p_configure)
    _add_work_queue_args(p_configure)
    _add_cli_input_args(p_configure)

    p_execute = sub.add_parser("execute", help="Execute from steps.yaml")
    p_execute.add_argument("--output", type=str, required=True)
    p_execute.add_argument("--steps", type=str, default="all")
    p_execute.add_argument(
        "--skip-refold",
        "--skip-af3-refold",
        dest="skip_refold",
        action="store_true",
        help="Skip AF3 refold + DockQ and force ranking to use AF3Score R2 metrics/structures.",
    )
    p_execute.add_argument("--reuse", action="store_true")
    p_execute.add_argument("--continue-on-error", action="store_true")
    p_execute.add_argument("--verbose", action="store_true")
    p_execute.add_argument("--num-devices", type=str, default=None, help="Number of GPUs/workers (e.g. 4 or 'all')")
    p_execute.add_argument("--devices", type=str, default=None, help="Comma-separated GPU list or 'all'")
    _add_orchestrator_pool_args(p_execute)
    p_execute.add_argument(
        "--single-process",
        action="store_true",
        help="Force legacy single-process execution (debug/regression only; disables orchestrator pools)",
    )
    _add_work_queue_args(p_execute)
    _add_run_lock_args(p_execute)

    p_rank = sub.add_parser("rank", help="Run rank step only")
    p_rank.add_argument("--output", type=str, required=True)
    p_rank.add_argument("--reuse", action="store_true")
    p_rank.add_argument(
        "--skip-refold",
        "--skip-af3-refold",
        dest="skip_refold",
        action="store_true",
        help="Force ranking to use AF3Score R2 metrics/structures (ignore refold artifacts).",
    )
    p_rank.add_argument("--continue-on-error", action="store_true")
    p_rank.add_argument("--verbose", action="store_true")
    _add_work_queue_args(p_rank)
    _add_run_lock_args(p_rank)

    p_orch = sub.add_parser("orchestrate", help="Run per-step work-queue pools")
    p_orch.add_argument("--output", type=str, required=True)
    p_orch.add_argument("--input", type=str, help="Path to design.yaml (used with --configure)")
    p_orch.add_argument("--preset", type=str, choices=["fast", "full", "custom"], default="full")
    p_orch.add_argument(
        "--output-mode",
        type=str,
        choices=["minimal", "full"],
        default=None,
        help="Output mode (minimal or full). Default: minimal.",
    )
    p_orch.add_argument("--configure", action="store_true", help="Run configure if steps.yaml is missing")
    p_orch.add_argument("--steps", type=str, default="all", help="Step names (all or comma-separated step list)")
    p_orch.add_argument(
        "--skip-refold",
        "--skip-af3-refold",
        dest="skip_refold",
        action="store_true",
        help="Skip AF3 refold + DockQ and force ranking to use AF3Score R2 metrics/structures.",
    )
    p_orch.add_argument("--pool-size", type=int, default=None, help="Override pool size (advanced)")
    p_orch.add_argument("--num-devices", type=str, default=None, help="Number of GPUs/workers (e.g. 4 or 'all')")
    _add_orchestrator_pool_args(p_orch)
    p_orch.add_argument("--max-retries", type=int, default=None, help="Max attempts per step")
    p_orch.add_argument(
        "--failure-policy",
        type=str,
        choices=["allow", "strict", "threshold"],
        default=None,
        help="Failure policy for item failures",
    )
    p_orch.add_argument("--no-bind", action="store_true", help="Do not set CUDA_VISIBLE_DEVICES or rank env vars")
    p_orch.add_argument("--devices", type=str, default=None, help="Comma-separated GPU list or 'all'")
    _add_run_lock_args(p_orch)

    sub.add_parser("wizard", help="Interactive setup wizard")

    return parser


def _single_process_conflicts(args) -> bool:
    return bool(getattr(args, "num_devices", None)) or bool(getattr(args, "devices", None)) or (
        getattr(args, "num_rosetta_workers", None) is not None or getattr(args, "num_cpu_workers", None) is not None
    )


def _apply_skip_refold_args(args) -> None:
    """
    Enforce --skip-refold semantics consistently across configure/execute/orchestrate.
    """
    if not getattr(args, "skip_refold", False):
        return

    conflicts = steps_conflict_with_skip_refold(getattr(args, "steps", None))
    if conflicts:
        raise SystemExit(f"--skip-refold conflicts with --steps containing: {', '.join(conflicts)}")

    raw_steps = getattr(args, "steps", None)
    raw_steps = "all" if raw_steps is None else str(raw_steps).strip()
    if not raw_steps or raw_steps.lower() == "all":
        effective = remove_skip_refold_steps(list(STEP_ORDER))
        args.steps = ",".join(effective)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _apply_skip_refold_args(args)

    # Defense-in-depth: orchestrator-spawned worker processes must never re-enter orchestration.
    # The orchestrator sets PPIFLOW_ORCH_WORKER=1 in the worker env; workers must use --single-process.
    if args.command == "execute" and os.environ.get("PPIFLOW_ORCH_WORKER") and not getattr(args, "single_process", False):
        raise SystemExit(
            "Internal error: orchestrator worker invoked without --single-process. "
            "This would cause recursive orchestration. Re-run with --single-process."
        )

    if args.command == "pipeline":
        out_dir = Path(args.output).resolve()
        steps_yaml = out_dir / "steps.yaml"
        config_dir = out_dir / "config"
        input_json = out_dir / "pipeline_input.json"
        missing_config = not steps_yaml.exists() or not config_dir.exists() or not input_json.exists()

        if getattr(args, "force_config", False):
            configure_pipeline(args)
        elif missing_config:
            if getattr(args, "skip_config", False):
                print("[pipeline] config files missing; running configure first.")
            configure_pipeline(args)
        else:
            if getattr(args, "input", None) and not getattr(args, "skip_config", False):
                print("[pipeline] using existing config; use --force-config to regenerate.")

        if getattr(args, "single_process", False):
            if _single_process_conflicts(args):
                raise SystemExit(
                    "--single-process cannot be combined with --num-devices/--devices or --num-rosetta-workers/--num-cpu-workers"
                )
            execute_pipeline(args)
        else:
            orchestrate_pipeline(args)
    elif args.command == "configure":
        configure_pipeline(args)
    elif args.command == "execute":
        if getattr(args, "single_process", False):
            if _single_process_conflicts(args):
                raise SystemExit(
                    "--single-process cannot be combined with --num-devices/--devices or --num-rosetta-workers/--num-cpu-workers"
                )
            execute_pipeline(args)
        else:
            orchestrate_pipeline(args)
    elif args.command == "rank":
        if getattr(args, "num_devices", None):
            raise SystemExit("rank does not support --num-devices (use execute/orchestrate instead)")
        args.steps = "rank_features,rank_finalize"
        execute_pipeline(args)
    elif args.command == "orchestrate":
        orchestrate_pipeline(args)
    elif args.command == "wizard":
        run_wizard()
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
