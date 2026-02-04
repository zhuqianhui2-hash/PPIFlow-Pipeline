from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .configure import configure_pipeline
from .execute import execute_pipeline
from .orchestrate import orchestrate_pipeline
from .io import repo_root


def _prompt_required(label: str) -> str:
    while True:
        value = input(label).strip()
        if value:
            return value
        print("Value is required.")


def _prompt_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value if value else default


def _prompt_optional(label: str) -> Optional[str]:
    value = input(label).strip()
    return value or None


def _prompt_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except Exception:
            print("Please enter a whole number.")


def _prompt_float(label: str) -> Optional[float]:
    raw = input(f"{label} (blank for default): ").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        print("Invalid number; leaving as default.")
        return None


def _prompt_yes_no(label: str, default_yes: bool = False) -> bool:
    hint = "Y/n" if default_yes else "y/N"
    raw = input(f"{label} ({hint}): ").strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"}


def _framework_choices() -> Dict[str, List[Dict[str, str]]]:
    root = repo_root()
    base = root / "assets" / "frameworks"
    vhh = [
        {
            "name": "5JDS nanobody",
            "path": str(base / "5jds_nanobody_framework.pdb"),
            "heavy_chain": "A",
            "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,21-21",
        },
        {
            "name": "7EOW nanobody",
            "path": str(base / "7eow_nanobody_framework.pdb"),
            "heavy_chain": "A",
            "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,20-20",
        },
        {
            "name": "7XL0 nanobody",
            "path": str(base / "7xl0_nanobody_framework.pdb"),
            "heavy_chain": "A",
            "cdr_length": "CDRH1,8-8,CDRH2,7-7,CDRH3,15-15",
        },
        {
            "name": "8COH nanobody",
            "path": str(base / "8coh_nanobody_framework.pdb"),
            "heavy_chain": "A",
            "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,19-19",
        },
        {
            "name": "8Z8V nanobody",
            "path": str(base / "8z8v_nanobody_framework.pdb"),
            "heavy_chain": "A",
            "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,8-8",
        },
    ]
    scfv = [
        {
            "name": "6NOU scFv",
            "path": str(base / "6nou_scfv_framework.pdb"),
            "heavy_chain": "A",
            "light_chain": "B",
            "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,12-12,CDRL1,11-11,CDRL2,3-3,CDRL3,9-9",
        },
        {
            "name": "6TCS scFv",
            "path": str(base / "6tcs_scfv_framework.pdb"),
            "heavy_chain": "A",
            "light_chain": "B",
            "cdr_length": "CDRH1,9-9,CDRH2,7-7,CDRH3,14-14,CDRL1,10-10,CDRL2,3-3,CDRL3,9-9",
        },
        {
            "name": "6ZQK scFv",
            "path": str(base / "6zqk_scfv_framework.pdb"),
            "heavy_chain": "A",
            "light_chain": "B",
            "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,13-13,CDRL1,6-6,CDRL2,3-3,CDRL3,9-9",
        },
    ]
    return {"vhh": vhh, "antibody": scfv}


def _select_framework(mode: str) -> Dict[str, str]:
    choices = _framework_choices()[mode]
    print("\nSelect framework:")
    for idx, item in enumerate(choices, start=1):
        if mode == "vhh":
            chain_info = f"(heavy={item['heavy_chain']})"
        else:
            chain_info = f"(heavy={item['heavy_chain']} light={item['light_chain']})"
        print(f"  {idx}) {item['name']} {chain_info}")
    print(f"  {len(choices) + 1}) Provide custom framework...")
    selection = _prompt_default("Choose", str(1))
    try:
        sel_idx = int(selection)
    except Exception:
        sel_idx = 1
    if sel_idx < 1:
        sel_idx = 1
    if sel_idx <= len(choices):
        return dict(choices[sel_idx - 1])
    # custom
    framework_pdb = _prompt_required("Framework PDB path: ")
    heavy_chain = _prompt_required("Heavy chain ID: ").strip()
    light_chain = None
    if mode == "antibody":
        light_chain = _prompt_required("Light chain ID: ").strip()
    cdr_length = _prompt_required("CDR length spec (e.g. CDRH1,8-8,...): ").strip()
    out = {"path": framework_pdb, "heavy_chain": heavy_chain, "cdr_length": cdr_length}
    if light_chain:
        out["light_chain"] = light_chain
    return out


def _build_args() -> argparse.Namespace:
    fields: Dict[str, Any] = {
        "output": None,
        "input": None,
        "preset": "full",
        "output_mode": None,
        "protocol": None,
        "steps": "all",
        "reuse": False,
        "continue_on_error": False,
        "verbose": False,
        "skip_config": False,
        "force_config": False,
        "num_devices": None,
        "devices": None,
        "no_bind": False,
        "pool_size": None,
        "max_retries": None,
        "failure_policy": None,
        "work_queue": True,
        "work_queue_lease_seconds": None,
        "work_queue_max_attempts": None,
        "work_queue_batch_size": None,
        "work_queue_leader_timeout": None,
        "work_queue_wait_timeout": None,
        "retry_failed": False,
        "work_queue_reuse": False,
        "work_queue_strict": False,
        "work_queue_rebuild": False,
        "af3_num_workers": None,
        # CLI-only input fields
        "name": None,
        "target_pdb": None,
        "target_chains": None,
        "hotspots": None,
        "binder_length": None,
        "samples_per_target": 100,
        "partial_samples_per_target": None,
        "framework_pdb": None,
        "heavy_chain": None,
        "light_chain": None,
        "cdr_length": None,
        "seq1_num_per_backbone": None,
        "seq1_temp": None,
        "seq1_bias_large_residues": False,
        "seq1_bias_num": None,
        "seq1_bias_residues": None,
        "seq1_bias_weight": None,
        "seq2_num_per_backbone": None,
        "seq2_temp": None,
        "seq2_use_soluble_ckpt": False,
        "vhh_backbones": None,
        "vhh_cdr1_num": None,
        "vhh_partial_num": None,
        "vhh_cdr2_num": None,
        "af3score1_iptm_min": None,
        "af3score1_ptm_min": None,
        "af3score1_top_k": None,
        "af3score2_iptm_min": None,
        "af3score2_ptm_min": None,
        "af3score2_top_k": None,
        "af3refold_iptm_min": None,
        "af3refold_ptm_min": None,
        "af3refold_dockq_min": None,
        "af3refold_num_samples": None,
        "af3refold_model_seeds": None,
        "af3refold_no_templates": None,
        "interface_energy_min": None,
        "interface_distance": None,
        "relax_max_iter": None,
        "relax_fixbb": None,
        "fixed_chains": None,
        "dockq_min": None,
        "partial_start_t": None,
        "rank_top_k": None,
        "af3_refold": False,
        "ppiflow_ckpt": None,
        "abmpnn_ckpt": None,
        "mpnn_ckpt": None,
        "mpnn_ckpt_soluble": None,
        "af3score_repo": None,
        "rosetta_bin": None,
        "rosetta_db": None,
        "flowpacker_repo": None,
        "af3_weights": None,
        "mpnn_repo": None,
        "abmpnn_repo": None,
        "mpnn_run": None,
        "abmpnn_run": None,
        "dockq_bin": None,
        "target_pdb_path": None,
        "target_chains_raw": None,
        "hotspots_raw": None,
        "target_pdb_full": None,
        "target_chains_full": None,
        "hotspots_full": None,
        "target_pdb_file": None,
        "target_chains_str": None,
        "hotspots_str": None,
        "target_chain_list": None,
    }
    return SimpleNamespace(**fields)


def run_wizard() -> None:
    if not sys.stdin.isatty():
        raise SystemExit("Wizard requires an interactive TTY.")

    print("\nPPIFlow Interactive Setup\n")

    project = _prompt_required("Project name: ")
    default_out = str(Path.cwd() / "runs" / project)
    out_dir = _prompt_default("Output directory (will create if missing)", default_out)
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    target_pdb = _prompt_required("Target PDB path: ")
    while not Path(target_pdb).expanduser().exists():
        print("Target PDB not found. Please provide a valid path.")
        target_pdb = _prompt_required("Target PDB path: ")

    target_chains = _prompt_required("Target chains (comma-separated): ")
    hotspots = _prompt_optional("Hotspots (optional, e.g. A3,A5-25,B72,B75): ")

    print("\nDesign mode:")
    print("  1) Binder")
    print("  2) Antibody (scFv)")
    print("  3) Nanobody (VHH)")
    mode_choice = _prompt_default("Select", "1")
    protocol = "binder"
    if mode_choice.strip() == "2":
        protocol = "antibody"
    elif mode_choice.strip() == "3":
        protocol = "vhh"

    binder_length = None
    framework_pdb = None
    heavy_chain = None
    light_chain = None
    cdr_length = None

    if protocol == "binder":
        binder_length = _prompt_default("Binder length (e.g. 75-90)", "75-90")
    else:
        framework = _select_framework("vhh" if protocol == "vhh" else "antibody")
        framework_pdb = framework["path"]
        heavy_chain = framework.get("heavy_chain")
        light_chain = framework.get("light_chain")
        cdr_length = framework.get("cdr_length")

    samples = _prompt_int("Samples per target", 100)

    print("\nGPU mode:")
    print("  1) single GPU")
    print("  2) all GPUs")
    print("  3) choose specific GPUs")
    gpu_choice = _prompt_default("Select", "1")
    num_devices = "1"
    devices = None
    if gpu_choice.strip() == "2":
        num_devices = "all"
    elif gpu_choice.strip() == "3":
        devices = _prompt_required("Enter GPUs (e.g. 0,2,3,6): ")
        device_list = [d.strip() for d in devices.split(",") if d.strip()]
        if not device_list:
            raise SystemExit("No GPUs provided.")
        num_devices = str(len(device_list))
        devices = ",".join(device_list)

    advanced = _prompt_yes_no("Advanced settings?", default_yes=False)
    af3score1_iptm_min = None
    af3score2_iptm_min = None
    af3score2_ptm_min = None
    dockq_min = None
    interface_energy_min = None
    interface_distance = None
    relax_max_iter = None
    rank_top_k = None
    if advanced:
        af3score1_iptm_min = _prompt_float("AF3Score round1 ipTM min")
        af3score2_iptm_min = _prompt_float("AF3Score round2 ipTM min")
        af3score2_ptm_min = _prompt_float("AF3Score round2 pTM min")
        dockq_min = _prompt_float("DockQ min")
        interface_energy_min = _prompt_float("Interface energy min (Rosetta)")
        interface_distance = _prompt_float("Interface distance (A)")
        relax_max_iter = _prompt_float("Relax max iterations")
        rank_top_k = _prompt_float("Rank top K")

    print("\nSummary:")
    print(f"  protocol: {protocol}")
    print(f"  name: {project}")
    print(f"  target: {target_pdb} (chains {target_chains})")
    if hotspots:
        print(f"  hotspots: {hotspots}")
    if protocol == "binder":
        print(f"  binder_length: {binder_length}")
    else:
        print(f"  framework: {framework_pdb} ({heavy_chain}{'/' + light_chain if light_chain else ''})")
        print(f"  cdr_length: {cdr_length}")
    print(f"  samples_per_target: {samples}")
    if devices:
        print(f"  num_devices: {num_devices} (GPUs: {devices})")
    else:
        print(f"  num_devices: {num_devices}")

    if not _prompt_yes_no("Run now?", default_yes=True):
        print("Exiting (no files written).")
        return

    args = _build_args()
    args.output = str(out_path)
    args.protocol = protocol
    args.name = project
    args.target_pdb = str(Path(target_pdb).expanduser())
    args.target_chains = target_chains
    args.hotspots = hotspots
    args.binder_length = binder_length
    args.framework_pdb = framework_pdb
    args.heavy_chain = heavy_chain
    args.light_chain = light_chain
    args.cdr_length = cdr_length
    args.samples_per_target = samples
    args.af3score1_iptm_min = af3score1_iptm_min
    args.af3score2_iptm_min = af3score2_iptm_min
    args.af3score2_ptm_min = af3score2_ptm_min
    args.dockq_min = dockq_min
    args.interface_energy_min = interface_energy_min
    args.interface_distance = interface_distance
    args.relax_max_iter = int(relax_max_iter) if isinstance(relax_max_iter, (int, float)) else None
    args.rank_top_k = int(rank_top_k) if isinstance(rank_top_k, (int, float)) else None
    args.num_devices = num_devices
    args.devices = devices

    configure_pipeline(args)
    if args.num_devices:
        orchestrate_pipeline(args)
    else:
        execute_pipeline(args)
