from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import PDB
from Bio.SeqUtils import seq1

from .base import Step, StepContext, StepError
from ..work_queue import WorkItem
from ..logging_utils import log_command_progress, run_command
from ..io import repo_root, is_ignored_path
from ..manifests import extract_design_id, structure_id_from_name, write_csv
from ..output_policy import is_minimal, scratch_dir


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p


def _chain_sequences(pdb_path: Path) -> dict[str, str]:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("partial", str(pdb_path))
    seqs: dict[str, str] = {}
    for chain in structure[0]:
        residues = []
        for res in chain:
            if PDB.is_aa(res, standard=True):
                residues.append(seq1(res.get_resname()))
        seqs[chain.id] = "".join(residues)
    return seqs


def _swap_pdb_chains(pdb_path: Path, mapping: dict[str, str]) -> None:
    if not mapping:
        return
    lines = pdb_path.read_text().splitlines()
    out_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM", "TER")) and len(line) > 21:
            chain_id = line[21]
            new_id = mapping.get(chain_id, chain_id)
            if new_id != chain_id:
                line = f"{line[:21]}{new_id}{line[22:]}"
        out_lines.append(line)
    pdb_path.write_text("\n".join(out_lines) + "\n")


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


def _apply_target_offsets(out_sub: Path, target_chain: str | None, mapping: list[int] | None) -> None:
    if not target_chain or not mapping:
        return
    for pdb_path in sorted(out_sub.rglob("sample*.pdb")):
        if not _renumber_chain_with_offsets(pdb_path, target_chain, mapping):
            raise StepError(f"Failed to apply target chain offsets to {pdb_path}")


def _fix_binder_partial_chains(out_sub: Path, target_pdb: Path, target_chain: str) -> None:
    if not out_sub.exists() or not target_pdb.exists() or not target_chain:
        return
    try:
        target_seq = _chain_sequences(target_pdb).get(target_chain)
    except Exception:
        target_seq = None
    if not target_seq:
        return
    for pdb_path in sorted(out_sub.rglob("sample*.pdb")):
        try:
            seqs = _chain_sequences(pdb_path)
        except Exception:
            continue
        if len(seqs) < 2:
            raise StepError(f"Partial flow output has <2 chains: {pdb_path}")
        if target_chain in seqs and seqs.get(target_chain) == target_seq:
            continue
        match_chain = None
        for cid, seq in seqs.items():
            if seq == target_seq:
                match_chain = cid
                break
        if not match_chain or match_chain == target_chain:
            raise StepError(f"Target chain sequence not found in partial output: {pdb_path}")
        mapping = {match_chain: target_chain}
        if target_chain in seqs:
            mapping[target_chain] = match_chain
        _swap_pdb_chains(pdb_path, mapping)


def _resolve_partial_pdb_path(ctx: StepContext, pdb_path: str | Path | None, row: dict[str, Any]) -> str | None:
    if pdb_path:
        candidate = Path(str(pdb_path))
        if not candidate.is_absolute():
            candidate = ctx.out_dir / candidate
        if candidate.exists():
            return str(candidate)
    name = row.get("pdb_name") or row.get("structure_id")
    if name:
        alt = ctx.out_dir / "output" / "rosetta_interface" / "rosetta_jobs" / str(name) / f"{name}.pdb"
        if alt.exists():
            return str(alt)
    return str(pdb_path) if pdb_path else None


def _resolve_ctx_path(ctx: StepContext, value: str | Path | None) -> Path | None:
    if not value:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (ctx.out_dir / p).resolve()


class PartialFlowStep(Step):
    name = "partial"
    stage = "partial"
    supports_indices = True
    supports_work_queue = True
    work_queue_mode = "items"

    def expected_total(self, ctx: StepContext) -> int:
        rows = self._load_fixed_positions(ctx)
        if not rows:
            raise StepError("No fixed positions found for partial flow")
        return len(rows)

    def _load_fixed_positions(self, ctx: StepContext) -> list[dict[str, Any]]:
        fixed_positions_csv = self.cfg.get("fixed_positions_csv")
        if not fixed_positions_csv:
            return []
        p = Path(fixed_positions_csv)
        if not p.is_absolute():
            p = ctx.out_dir / p
        if not p.exists():
            return []
        df = pd.read_csv(p)
        rows = df.to_dict(orient="records")
        return rows

    def scan_done(self, ctx: StepContext) -> set[int]:
        out_dir = self.output_dir(ctx)
        rows = self._load_fixed_positions(ctx)
        done: set[int] = set()
        for idx, row in enumerate(rows):
            sid = str(row.get("structure_id") or row.get("pdb_name") or idx)
            sub = out_dir / sid
            if sub.exists():
                if list(sub.rglob("sample*.pdb")) or list(sub.rglob("sample*_*.pdb")):
                    done.add(idx)
        return done

    def run_indices(self, ctx: StepContext, indices: list[int]) -> None:
        rows = self._load_fixed_positions(ctx)
        if not rows:
            raise StepError("fixed_positions_csv missing or empty")
        protocol = ctx.input_data.get("protocol")
        out_dir = self.output_dir(ctx)
        name = ctx.input_data.get("name")
        tools = ctx.input_data.get("tools") or {}
        partial_cfg = ctx.input_data.get("partial") or {}
        samples_per_target = int(partial_cfg.get("samples_per_target") or 8)

        if protocol == "binder":
            allow_failures = bool((ctx.input_data.get("options") or {}).get("continue_on_item_error"))
            script = _resolve_repo_path("src/entrypoints/sample_binder_partial.py")
            config_path = tools.get("ppiflow_binder_partial_config") or str(
                _resolve_repo_path("src/configs/inference_binder_partial.yaml")
            )
            config_path = str(_resolve_repo_path(config_path))
            target = ctx.input_data.get("target") or {}
            binder_chain = ctx.input_data.get("binder_chain") or "A"
            target_chain = target.get("chains", [None])[0]
            offsets_path = _resolve_ctx_path(ctx, target.get("chain_offsets"))
            offset_map = None
            if offsets_path:
                if not offsets_path.exists():
                    raise StepError(f"Target offsets file not found: {offsets_path}")
                offset_map = _load_chain_offset_map(offsets_path)
                if not offset_map:
                    raise StepError(f"Failed to load target chain offsets: {offsets_path}")
            ckpt = tools.get("ppiflow_ckpt")
            if not ckpt:
                raise StepError("tools.ppiflow_ckpt is required for binder partial flow")
            total = len(indices)
            failures: list[tuple[str, str]] = []
            for pos, idx in enumerate(indices, start=1):
                row = rows[idx]
                sid = str(row.get("structure_id") or row.get("pdb_name") or idx)
                pdb_path = _resolve_partial_pdb_path(ctx, row.get("pdb_path") or target.get("pdb"), row)
                motif_contig = _normalize_optional_string(row.get("motif_contig"))
                out_sub = out_dir / sid
                cmd = [
                    sys.executable,
                    str(script),
                    "--input_pdb",
                    str(pdb_path),
                    "--target_chain",
                    str(target_chain),
                    "--binder_chain",
                    str(binder_chain),
                    "--config",
                    str(config_path),
                    "--samples_per_target",
                    str(samples_per_target),
                    "--model_weights",
                    str(ckpt),
                    "--output_dir",
                    str(out_sub),
                    "--name",
                    str(name),
                    "--seed",
                    str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0)),
                ]
                if motif_contig:
                    cmd.extend(["--motif_contig", str(motif_contig)])
                hotspots = target.get("hotspots")
                hotspots_file = target.get("hotspots_file")
                if hotspots_file:
                    cmd.extend(["--hotspots_file", str(hotspots_file)])
                elif hotspots:
                    if isinstance(hotspots, list):
                        cmd.extend(["--specified_hotspots", ",".join(hotspots)])
                    else:
                        cmd.extend(["--specified_hotspots", str(hotspots)])
                start = time.time()
                status = "OK"
                try:
                    env = os.environ.copy()
                    # Force single-process execution for Lightning to avoid DDP/torchrun inside workers.
                    env.update({
                        "RANK": "0",
                        "WORLD_SIZE": "1",
                        "LOCAL_RANK": "0",
                        "NODE_RANK": "0",
                        "MASTER_ADDR": "127.0.0.1",
                        "MASTER_PORT": "29500",
                    })
                    for key in ("TORCHRUN", "TORCHELASTIC_RUN_ID", "GROUP_RANK", "LOCAL_WORLD_SIZE"):
                        env.pop(key, None)
                    if is_minimal(ctx):
                        wandb_root = scratch_dir(ctx) / "wandb"
                        wandb_root.mkdir(parents=True, exist_ok=True)
                        env.setdefault("WANDB_MODE", "disabled")
                        env.setdefault("WANDB_DIR", str(wandb_root))
                        cmd.extend(["--wandb_mode", "disabled", "--wandb_dir", str(wandb_root)])
                    run_command(
                        cmd,
                        env=env,
                        log_file=self.cfg.get("_log_file"),
                        verbose=bool(self.cfg.get("_verbose")),
                    )
                    target_pdb = target.get("pdb")
                    if target_pdb and target_chain:
                        _fix_binder_partial_chains(out_sub, Path(str(target_pdb)), str(target_chain))
                    _apply_target_offsets(out_sub, str(target_chain) if target_chain else None, offset_map)
                except Exception as exc:
                    status = "FAILED"
                    if allow_failures:
                        failures.append((sid, str(exc)))
                    else:
                        raise
                finally:
                    log_command_progress(
                        str(self.cfg.get("name") or self.name),
                        pos,
                        total,
                        item=sid,
                        status=status,
                        elapsed=time.time() - start,
                        log_file=self.cfg.get("_log_file"),
                    )
            if failures:
                fail_path = out_dir / "partial_failures.txt"
                fail_path.write_text("\n".join(f"{sid}\t{err}" for sid, err in failures) + "\n")
                if len(failures) >= total:
                    raise StepError("partial flow failed for all items")
        else:
            allow_failures = bool((ctx.input_data.get("options") or {}).get("continue_on_item_error"))
            script = _resolve_repo_path("src/entrypoints/sample_antibody_nanobody_partial.py")
            config_path = tools.get("ppiflow_antibody_partial_config") or str(
                _resolve_repo_path("src/configs/inference_nanobody.yaml")
            )
            config_path = str(_resolve_repo_path(config_path))
            target = ctx.input_data.get("target") or {}
            framework = ctx.input_data.get("framework") or {}
            target_chain = (target.get("chains") or [None])[0]
            offsets_path = _resolve_ctx_path(ctx, target.get("chain_offsets"))
            offset_map = None
            if offsets_path:
                if not offsets_path.exists():
                    raise StepError(f"Target offsets file not found: {offsets_path}")
                offset_map = _load_chain_offset_map(offsets_path)
                if not offset_map:
                    raise StepError(f"Failed to load target chain offsets: {offsets_path}")
            ckpt = tools.get("ppiflow_ckpt")
            if not ckpt:
                raise StepError("tools.ppiflow_ckpt is required for antibody/vhh partial flow")
            total = len(indices)
            failures: list[tuple[str, str]] = []
            for pos, idx in enumerate(indices, start=1):
                row = rows[idx]
                sid = str(row.get("structure_id") or row.get("pdb_name") or idx)
                pdb_path = _resolve_partial_pdb_path(ctx, row.get("pdb_path"), row)
                if not pdb_path:
                    raise StepError("fixed_positions.csv missing pdb_path for partial flow")
                if "fixed_positions" not in row:
                    raise StepError("fixed_positions.csv missing fixed_positions")
                fixed_positions = _normalize_optional_string(row.get("fixed_positions")) or ""
                out_sub = out_dir / sid
                cmd = [
                    sys.executable,
                    str(script),
                    "--complex_pdb",
                    str(pdb_path),
                    "--fixed_positions",
                    str(fixed_positions),
                    "--antigen_chain",
                    ",".join(target.get("chains") or []),
                    "--heavy_chain",
                    str(framework.get("heavy_chain")),
                    "--samples_per_target",
                    str(samples_per_target),
                    "--config",
                    str(config_path),
                    "--model_weights",
                    str(ckpt),
                    "--output_dir",
                    str(out_sub),
                    "--name",
                    str(name),
                    "--start_t",
                    str(ctx.input_data.get("partial", {}).get("start_t", 0.6)),
                    "--seed",
                    str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0)),
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
                start = time.time()
                status = "OK"
                try:
                    env = os.environ.copy()
                    env.update({
                        "RANK": "0",
                        "WORLD_SIZE": "1",
                        "LOCAL_RANK": "0",
                        "NODE_RANK": "0",
                        "MASTER_ADDR": "127.0.0.1",
                        "MASTER_PORT": "29500",
                    })
                    for key in ("TORCHRUN", "TORCHELASTIC_RUN_ID", "GROUP_RANK", "LOCAL_WORLD_SIZE"):
                        env.pop(key, None)
                    if is_minimal(ctx):
                        wandb_root = scratch_dir(ctx) / "wandb"
                        wandb_root.mkdir(parents=True, exist_ok=True)
                        env.setdefault("WANDB_MODE", "disabled")
                        env.setdefault("WANDB_DIR", str(wandb_root))
                        cmd.extend(["--wandb_mode", "disabled", "--wandb_dir", str(wandb_root)])
                    run_command(
                        cmd,
                        env=env,
                        log_file=self.cfg.get("_log_file"),
                        verbose=bool(self.cfg.get("_verbose")),
                    )
                    _apply_target_offsets(out_sub, str(target_chain) if target_chain else None, offset_map)
                except Exception as exc:
                    status = "FAILED"
                    if allow_failures:
                        failures.append((sid, str(exc)))
                    else:
                        raise
                finally:
                    log_command_progress(
                        str(self.cfg.get("name") or self.name),
                        pos,
                        total,
                        item=sid,
                        status=status,
                        elapsed=time.time() - start,
                        log_file=self.cfg.get("_log_file"),
                    )
            if failures:
                fail_path = out_dir / "partial_failures.txt"
                fail_path.write_text("\n".join(f"{sid}\t{err}" for sid, err in failures) + "\n")
                if len(failures) >= total:
                    raise StepError("partial flow failed for all items")

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        rows = []
        for fp in out_dir.rglob("sample*.pdb"):
            if is_ignored_path(fp):
                continue
            stem = fp.stem
            did = extract_design_id(stem) or extract_design_id(fp.parent.name)
            rows.append({
                "design_id": did,
                "structure_id": structure_id_from_name(fp.parent.name),
                "pdb_path": str(fp),
            })
        if not rows:
            return
        write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "pdb_path"])

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        rows = self._load_fixed_positions(ctx)
        items: list[WorkItem] = []
        for idx, row in enumerate(rows):
            sid = str(row.get("structure_id") or row.get("pdb_name") or idx)
            items.append(WorkItem(id=str(idx), payload={"row_idx": idx, "sid": sid}))
        return items

    def item_done(self, ctx: StepContext, item: WorkItem) -> bool:
        out_dir = self.output_dir(ctx)
        sid = str((item.payload or {}).get("sid"))
        sub = out_dir / sid
        if sub.exists():
            if list(sub.rglob("sample*.pdb")) or list(sub.rglob("sample*_*.pdb")):
                return True
        return False

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        row_idx = int((item.payload or {}).get("row_idx"))
        self.run_indices(ctx, [row_idx])
