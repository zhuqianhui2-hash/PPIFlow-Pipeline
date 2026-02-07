from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .base import Step, StepContext, StepError
from ..direct_legacy import compute_run_stems, files_identical, promote_file, promote_file_atomic, promote_tree
from ..io import collect_pdbs, is_ignored_path
from ..logging_utils import log_command_progress, run_command
from ..manifests import build_name_map, extract_design_id, find_metrics_file, structure_id_from_name, write_csv
from ..output_policy import is_minimal, mode as output_mode, optional_dir, should_keep, step_scratch_dir
from ..work_queue import WorkItem


def _norm_metrics_desc(value: str) -> str:
    norm = str(value or "").strip().lower()
    if norm.endswith(".pdb"):
        norm = norm[:-4]
    return norm


def is_valid_metrics_shard(path: Path, *, expected_desc: str | None = None) -> bool:
    """
    Cheap per-item metrics CSV validity check.

    We intentionally avoid pandas here to keep it fast and resilient to partially-written/empty files.
    """
    try:
        if not path.exists():
            return False
        # Reject newline-only and other trivially broken files early.
        if path.stat().st_size < 20:
            return False
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return False
            lower_map = {str(n).strip().lower(): n for n in reader.fieldnames}
            desc_col = None
            for key in ("description", "name", "model", "pdb_name"):
                col = lower_map.get(key)
                if col is not None:
                    desc_col = col
                    break
            if desc_col is None:
                return False
            total_rows = 0
            matched = 0
            expected_norm = _norm_metrics_desc(expected_desc) if expected_desc is not None else None
            # Per-item shards should contain exactly one row. We only scan a small prefix
            # to keep this check cheap.
            for row in reader:
                total_rows += 1
                if expected_norm is not None:
                    if _norm_metrics_desc(row.get(desc_col) or "") == expected_norm:
                        matched += 1
                if total_rows >= 10:
                    break
            if total_rows == 0:
                return False
            if expected_norm is not None:
                # Require exactly one matching row and no extra rows; this prevents
                # accidentally accepting multi-item or corrupted shards.
                if matched != 1 or total_rows != 1:
                    return False
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class _AF3JobSource:
    kind: str  # "seed" or "root"
    cif_path: Path
    summary_conf: Path
    full_conf: Path


def _parseable_json(path: Path) -> bool:
    try:
        payload = path.read_text()
    except Exception:
        return False
    try:
        json.loads(payload)
    except Exception:
        return False
    return True


def _best_seed_model_cif(seed_dir: Path) -> Path | None:
    cand = seed_dir / "model.cif"
    if cand.exists():
        return cand
    nested = sorted(seed_dir.glob("*/model.cif"))
    if nested:
        return nested[0]
    return None


_SEED_DIR_RE = re.compile(r"^seed-(?P<seed>\\d+)(?:_sample-(?P<sample>\\d+))?$")


def _seed_sort_key(path: Path) -> tuple[int, int, str]:
    m = _SEED_DIR_RE.match(path.name)
    if not m:
        return (10**9, 10**9, path.name)
    seed = int(m.group("seed"))
    sample = int(m.group("sample") or 0)
    return (seed, sample, path.name)


def _pick_af3_job_source(job_dir: Path, job_name: str, *, prefer_cif: Path | None = None) -> _AF3JobSource | None:
    """
    Pick a single canonical source for both promoted structure artifacts and metrics extraction.

    Seed-only selection (Option A): only select a seed/sample directory that contains:
    - model.cif (or nested */model.cif)
    - summary_confidences.json (parseable)
    - confidences.json (non-empty)

    If prefer_cif is provided, only return sources whose model.cif is byte-identical to prefer_cif.
    """
    job_dir = Path(job_dir)
    job_name = str(job_name)

    # Prefer seed-*_{sample-*} first, then seed-* as a fallback.
    seed_dirs = [p for p in job_dir.glob("seed-*_sample-*") if p.is_dir()]
    if not seed_dirs:
        seed_dirs = [p for p in job_dir.glob("seed-*") if p.is_dir()]
    seed_dirs = sorted(seed_dirs, key=_seed_sort_key)
    for seed_dir in seed_dirs:
        summary = seed_dir / "summary_confidences.json"
        full = seed_dir / "confidences.json"
        if not summary.exists() or not full.exists():
            continue
        try:
            if full.stat().st_size < 20:
                continue
        except Exception:
            continue
        if not _parseable_json(summary):
            continue
        cif = _best_seed_model_cif(seed_dir)
        if cif is None or not cif.exists():
            continue
        if prefer_cif is not None:
            try:
                if not files_identical(cif, prefer_cif):
                    continue
            except Exception:
                continue
        return _AF3JobSource(kind="seed", cif_path=cif, summary_conf=summary, full_conf=full)
    return None


class ExternalCommandStep(Step):
    supports_indices = False
    supports_work_queue = True
    work_queue_mode = "leader"

    def expected_total(self, ctx: StepContext) -> int:
        # Default to number of samples
        sampling = ctx.input_data.get("sampling") or {}
        return int(sampling.get("samples_per_target", 0) or 0)

    def _manifest_has_rows(self, ctx: StepContext) -> bool:
        manifest = self.cfg.get("manifest")
        if not manifest:
            return False
        p = Path(manifest)
        if not p.is_absolute():
            p = ctx.out_dir / p
        if not p.exists():
            return False
        try:
            with p.open("r", newline="") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
            return len(rows) > 1
        except Exception:
            return False

    def scan_done(self, ctx: StepContext) -> set[int]:
        if self._manifest_has_rows(ctx):
            return set(range(self.expected_total(ctx)))
        return set()

    def run_full(self, ctx: StepContext) -> None:
        cmd = self.cfg.get("command")
        if not cmd:
            raise StepError(
                f"Step {self.name} missing command. Edit {self.cfg.get('config_path')} to provide a command."
            )
        if isinstance(cmd, str):
            cmd = [cmd]
        env = os.environ.copy()
        step_label = str(self.cfg.get("name") or self.name)
        start = time.time()
        status = "OK"
        try:
            run_command(
                cmd,
                env=env,
                log_file=self.cfg.get("_log_file"),
                verbose=bool(self.cfg.get("_verbose")),
            )
        except Exception:
            status = "FAILED"
            raise
        finally:
            log_command_progress(
                step_label,
                1,
                1,
                item="command",
                status=status,
                elapsed=time.time() - start,
                log_file=self.cfg.get("_log_file"),
            )


class SeqDesignStep(ExternalCommandStep):
    name = "seq"
    stage = "seq"
    supports_work_queue = True
    work_queue_mode = "items"
    # Drain per-worker batches to avoid repeated model reloads.
    per_worker_batch = True
    # 0 means "claim all available items for this worker".
    batch_size = 0

    def _default_chain_list(self, ctx: StepContext) -> str:
        protocol = ctx.input_data.get("protocol")
        if protocol == "binder":
            return str(ctx.input_data.get("binder_chain") or "A")
        framework = ctx.input_data.get("framework") or {}
        heavy = str(framework.get("heavy_chain") or "A")
        light = framework.get("light_chain")
        if light:
            return f"{heavy},{light}"
        return heavy

    def _default_mpnn_run(self, ctx: StepContext) -> Path | None:
        tools = ctx.input_data.get("tools") or {}
        if ctx.input_data.get("protocol") == "binder":
            return Path(tools.get("mpnn_run") or tools.get("mpnn_repo", "")).joinpath("protein_mpnn_run.py")
        return Path(tools.get("abmpnn_run") or tools.get("abmpnn_repo", "")).joinpath("protein_mpnn_run.py")

    def _resolve_weights(self, ckpt: str | None) -> tuple[Path | None, str | None]:
        if not ckpt:
            return None, None
        path = Path(ckpt)
        if path.is_file():
            return path.parent, path.stem
        return path, None

    def _write_bias_jsonl(self, out_dir: Path, residues: list[str], weight: float) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        bias = {r: float(weight) for r in residues}
        path = out_dir / "bias_residues.jsonl"
        path.write_text(json.dumps(bias) + "\n")
        return path

    def _parse_fixed_positions_by_chain(
        self, value: str, default_chain: str
    ) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = {}
        if not value:
            return mapping
        tokens = re.split(r"[\\s,]+", str(value).strip())
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            match = re.match(r"^([A-Za-z]+)?(\\d+)(?:-(?:[A-Za-z]+)?(\\d+))?$", token)
            if not match:
                continue
            chain = match.group(1) or default_chain or "A"
            start = int(match.group(2))
            end = int(match.group(3) or start)
            if end < start:
                start, end = end, start
            indices = list(range(start, end + 1))
            mapping.setdefault(chain, []).extend(indices)
        return {k: sorted(set(v)) for k, v in mapping.items()}

    def _build_fixed_positions_map(self, fixed_positions_csv: str | Path) -> dict[str, dict[str, object]]:
        mapping: dict[str, dict[str, object]] = {}
        path = Path(fixed_positions_csv)
        if not path.exists():
            return mapping
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sid = str(row.get("structure_id") or row.get("pdb_name") or "").strip()
                if not sid:
                    continue
                chain = str(row.get("binder_chain") or row.get("chain") or "").strip()
                indices_raw = (
                    row.get("fixed_sequence")
                    or row.get("fixed_positions_indices")
                    or row.get("fixed_positions")
                    or ""
                )
                fixed_positions = self._parse_fixed_positions_by_chain(indices_raw, chain or "A")
                entry = {"fixed_positions": fixed_positions}
                mapping[sid] = entry
                pname = str(row.get("pdb_name") or "").strip()
                if pname:
                    mapping.setdefault(pname, entry)
        return mapping

    def _resolve_fixed_positions(
        self,
        pdb_path: Path,
        mapping: dict[str, dict[str, object]],
    ) -> dict[str, object] | None:
        stem = pdb_path.stem
        if "__" in stem:
            sid = stem.split("__", 1)[0]
            if sid in mapping:
                return mapping[sid]
        if stem in mapping:
            return mapping[stem]
        parent = pdb_path.parent.name
        if parent in mapping:
            return mapping[parent]
        return None

    def _write_fixed_positions_jsonl(
        self,
        out_dir: Path,
        pdbs: list[Path],
        fixed_positions_csv: str | Path,
        default_chain: str,
    ) -> Path | None:
        mapping = self._build_fixed_positions_map(fixed_positions_csv)
        if not mapping:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "fixed_positions.jsonl"
        count = 0
        with jsonl_path.open("w") as handle:
            for pdb_path in pdbs:
                entry = self._resolve_fixed_positions(pdb_path, mapping)
                if not entry:
                    continue
                fixed_positions = entry.get("fixed_positions")
                if isinstance(fixed_positions, dict) and fixed_positions:
                    payload = {"name": pdb_path.stem, "fixed_positions": fixed_positions}
                else:
                    indices = [int(x) for x in entry.get("indices") or []]
                    if not indices:
                        continue
                    chain = str(entry.get("chain") or default_chain or "A")
                    payload = {
                        "name": pdb_path.stem,
                        "fixed_positions": {chain: indices},
                    }
                handle.write(json.dumps(payload) + "\n")
                count += 1
        if count == 0:
            return None
        return jsonl_path

    def _write_fixed_positions_jsonl_item(
        self,
        out_dir: Path,
        pdb_path: Path,
        run_stem: str,
        fixed_positions_csv: str | Path,
        default_chain: str,
    ) -> Path | None:
        mapping = self._build_fixed_positions_map(fixed_positions_csv)
        if not mapping:
            return None
        entry = self._resolve_fixed_positions(pdb_path, mapping)
        if not entry:
            return None
        fixed_positions = entry.get("fixed_positions")
        if isinstance(fixed_positions, dict) and fixed_positions:
            payload = {"name": run_stem, "fixed_positions": fixed_positions}
        else:
            indices = [int(x) for x in entry.get("indices") or []]
            if not indices:
                return None
            chain = str(entry.get("chain") or default_chain or "A")
            payload = {
                "name": run_stem,
                "fixed_positions": {chain: indices},
            }
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "fixed_positions.jsonl"
        jsonl_path.write_text(json.dumps(payload) + "\n")
        return jsonl_path

    def _fixed_positions_from_bfactor(
        self,
        pdb_path: Path,
        chains: list[str],
        threshold: float = 3.9,
    ) -> dict[str, list[int]]:
        positions: dict[str, set[int]] = {}
        chain_set = {c.strip() for c in chains if c and str(c).strip()}
        if not chain_set:
            return {}
        try:
            lines = pdb_path.read_text().splitlines()
        except Exception:
            return {}
        for line in lines:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 66:
                continue
            chain_id = line[21].strip()
            if chain_id not in chain_set:
                continue
            try:
                b_factor = float(line[60:66])
            except Exception:
                continue
            if b_factor < threshold:
                continue
            try:
                resseq = int(line[22:26])
            except Exception:
                continue
            positions.setdefault(chain_id, set()).add(resseq)
        return {c: sorted(v) for c, v in positions.items() if v}

    def _write_fixed_positions_jsonl_from_mapping(
        self,
        out_dir: Path,
        run_stem: str,
        mapping: dict[str, list[int]],
    ) -> Path | None:
        if not mapping:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"name": run_stem, "fixed_positions": mapping}
        jsonl_path = out_dir / "fixed_positions.jsonl"
        jsonl_path.write_text(json.dumps(payload) + "\n")
        return jsonl_path

    def _merge_fastas(self, vanilla_dir: Path, bias_dir: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        seq_dir = out_dir / "seqs"
        seq_dir.mkdir(parents=True, exist_ok=True)
        vanilla_seq = vanilla_dir / "seqs"
        bias_seq = bias_dir / "seqs"

        vanilla_files = sorted(vanilla_seq.glob("*.fa*")) if vanilla_seq.exists() else []
        bias_files = sorted(bias_seq.glob("*.fa*")) if bias_seq.exists() else []

        if not vanilla_files and not bias_files:
            return

        def _atomic_write_text(path: Path, content: str) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f".tmp.{path.name}.{os.getpid()}.{threading.get_ident()}"
            tmp.write_text(content)
            os.replace(tmp, path)

        for fasta in vanilla_files:
            name = fasta.name
            lines = []
            for line in fasta.read_text().splitlines():
                if line.startswith(">"):
                    line = f"{line}|vanilla"
                lines.append(line)
            bias_path = bias_seq / name
            if bias_path.exists():
                for line in bias_path.read_text().splitlines():
                    if line.startswith(">"):
                        line = f"{line}|bias"
                    lines.append(line)
            _atomic_write_text(seq_dir / name, "\n".join(lines) + "\n")
        # Handle bias-only files
        for fasta in bias_files:
            name = fasta.name
            if (seq_dir / name).exists():
                continue
            lines = []
            for line in fasta.read_text().splitlines():
                if line.startswith(">"):
                    line = f"{line}|bias"
                lines.append(line)
            _atomic_write_text(seq_dir / name, "\n".join(lines) + "\n")

    def _collect_pdbs(self, ctx: StepContext, input_dir: Path) -> list[Path]:
        pdbs = collect_pdbs(input_dir)
        protocol = ctx.input_data.get("protocol")
        if protocol in {"antibody", "vhh"} and str(self.cfg.get("name") or "") == "seq1":
            metrics_path = input_dir / "sample_metrics.csv"
            df = None
            if metrics_path.exists():
                try:
                    df = pd.read_csv(metrics_path)
                except Exception:
                    df = None
                else:
                    if "cdr_interface_ratio" not in df.columns or "pdb_path" not in df.columns:
                        df = None

            if df is not None:
                try:
                    df = df[df["cdr_interface_ratio"].astype(float) >= 0.6]
                except Exception:
                    df = df[df["cdr_interface_ratio"] >= 0.6]
                if df.empty:
                    raise StepError("No backbones pass cdr_interface_ratio >= 0.6")
                filtered: list[Path] = []
                for path_val in df["pdb_path"].tolist():
                    if not path_val:
                        continue
                    cand = Path(str(path_val))
                    if not cand.is_absolute():
                        cand = (input_dir / cand).resolve()
                    if cand.exists():
                        # Prefer paths under the input_dir to keep run_stem resolution stable.
                        try:
                            cand.relative_to(input_dir)
                        except Exception:
                            alt = input_dir / cand.name
                            if alt.exists():
                                cand = alt
                        filtered.append(cand)
                        continue
                    # Fallback for moved outputs: try input_dir/<name>.pdb
                    alt = input_dir / Path(str(path_val)).name
                    if alt.exists():
                        filtered.append(alt)
                        continue
                    alt = input_dir / f"{Path(str(path_val)).stem}.pdb"
                    if alt.exists():
                        filtered.append(alt)
                if not filtered:
                    raise StepError("No backbone PDBs found after cdr_interface_ratio prefilter")
                return filtered

            # Output-first robustness: if sample_metrics.csv is missing/corrupt, recompute the
            # single metric we need (cdr_interface_ratio) directly from the on-disk PDBs.
            framework = ctx.input_data.get("framework") or {}
            heavy_chain = str(framework.get("heavy_chain") or "A")
            light_chain = str(framework.get("light_chain") or "")
            try:
                from analysis.antibody_metric import get_interface_residues
            except Exception as exc:  # pragma: no cover
                raise StepError(
                    f"Missing or invalid {metrics_path} for CDR interface prefilter and failed to compute it from PDBs: {exc}"
                ) from exc

            filtered: list[Path] = []
            for pdb_path in pdbs:
                try:
                    _, _, ratio = get_interface_residues(str(pdb_path), 10, heavy_chain, light_chain)
                except Exception:
                    ratio = 0.0
                if float(ratio) >= 0.6:
                    filtered.append(pdb_path)
            if not filtered:
                raise StepError("No backbones pass cdr_interface_ratio >= 0.6")
            return filtered
        return pdbs

    def _stage_pdbs(self, pdbs: list[Path], out_dir: Path) -> list[Path]:
        if not pdbs:
            return []
        stems = [p.stem for p in pdbs]
        should_stage = self.cfg.get("name") == "seq2" or len(set(stems)) != len(stems)
        if not should_stage:
            return pdbs
        stage_dir = out_dir / "pdbs"
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged: list[Path] = []
        for pdb_path in pdbs:
            parent = pdb_path.parent.name
            unique_stem = f"{parent}__{pdb_path.stem}" if should_stage else pdb_path.stem
            dst = stage_dir / f"{unique_stem}.pdb"
            if not dst.exists():
                try:
                    os.link(pdb_path, dst)
                except OSError:
                    shutil.copy2(pdb_path, dst)
            staged.append(dst)
        return staged

    def _find_fasta(self, out_dir: Path, stem: str) -> Path | None:
        for base in [out_dir / "seqs", out_dir]:
            for suffix in [".fa", ".fasta", ".fa.gz", ".fasta.gz"]:
                cand = base / f"{stem}{suffix}"
                if cand.exists():
                    return cand
        return None

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        input_dir = self.cfg.get("input_dir")
        if not input_dir:
            raise StepError(f"Step {self.name} missing input_dir")
        input_path = Path(input_dir)
        if not input_path.is_absolute():
            input_path = ctx.out_dir / input_path
        if not input_path.exists():
            raise StepError(f"Input dir not found: {input_path}")
        pdbs = self._collect_pdbs(ctx, input_path)
        if not pdbs:
            raise StepError(f"No PDBs found for sequence design in {input_path}")
        run_stems = compute_run_stems(pdbs, input_path)

        position_list = self.cfg.get("fixed_positions_csv")
        chain_list = self._default_chain_list(ctx)
        default_chain = str(chain_list).split(",")[0].strip() or "A"

        items: list[WorkItem] = []
        for pdb_path in pdbs:
            items.append(
                WorkItem(
                    id=pdb_path.stem if run_stems[pdb_path] == pdb_path.stem else run_stems[pdb_path],
                    payload={
                        "pdb_path": str(pdb_path),
                        "orig_stem": pdb_path.stem,
                        "run_stem": run_stems[pdb_path],
                        "input_root": str(input_path),
                        "fixed_positions_csv": str(position_list) if position_list else None,
                        "default_chain": default_chain,
                    },
                )
            )
        return items

    def item_done(self, ctx: StepContext, item: WorkItem) -> bool:
        out_dir = self.output_dir(ctx)
        stem = str((item.payload or {}).get("run_stem") or (item.payload or {}).get("pdb_name") or item.id)
        step_name = str(self.cfg.get("name") or self.name)
        if step_name == "seq2":
            pdb_path = out_dir / "pdbs" / f"{stem}.pdb"
            if not pdb_path.exists():
                return False
        return self._find_fasta(out_dir, stem) is not None

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        tools = ctx.input_data.get("tools") or {}
        out_dir = self.output_dir(ctx)
        step_name = str(self.cfg.get("name") or self.name)
        if is_minimal(ctx):
            item_tmp = step_scratch_dir(ctx, step_name) / item.id
        else:
            item_tmp = out_dir / ".tmp" / item.id
        item_tmp.mkdir(parents=True, exist_ok=True)

        mpnn_run = self._default_mpnn_run(ctx)
        if not mpnn_run or not mpnn_run.exists():
            raise StepError(
                f"MPNN runner not found. Set tools.mpnn_run/abmpnn_run or provide command in {self.cfg.get('config_path')}"
            )
        ckpt = tools.get("mpnn_ckpt") if ctx.input_data.get("protocol") == "binder" else tools.get("abmpnn_ckpt")

        seq_cfg = (ctx.input_data.get("sequence_design") or {}).get(
            "round1" if self.cfg.get("name") == "seq1" else "round2"
        ) or {}
        num_seq = int(seq_cfg.get("num_seq_per_backbone") or 0)
        sampling_temp = float(seq_cfg.get("sampling_temp") or 0.1)
        if num_seq <= 0:
            raise StepError("sequence_design.num_seq_per_backbone must be > 0")
        chain_list = self._default_chain_list(ctx)
        protocol = ctx.input_data.get("protocol")
        omit_aas = seq_cfg.get("omit_aas") if protocol != "binder" else None
        use_soluble = bool(seq_cfg.get("use_soluble_ckpt"))
        if protocol == "binder" and use_soluble and self.cfg.get("name") == "seq2":
            ckpt = tools.get("mpnn_ckpt_soluble") or ckpt

        weight_dir, model_name = self._resolve_weights(ckpt)
        chain_arg = " ".join(str(chain_list).replace(",", " ").split())
        seed = str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0))
        fixed_positions_jsonl = None
        fixed_positions_csv = (item.payload or {}).get("fixed_positions_csv")
        run_stem = str((item.payload or {}).get("run_stem") or "")
        if fixed_positions_csv:
            default_chain = str((item.payload or {}).get("default_chain") or "A")
            fixed_positions_jsonl = self._write_fixed_positions_jsonl_item(
                item_tmp,
                Path(str((item.payload or {}).get("pdb_path") or "")),
                run_stem,
                fixed_positions_csv,
                default_chain,
            )

        def build_base_cmd(out_folder: Path, num_seq_target: int, temp: float, use_soluble_model: bool) -> list[str]:
            cmd = [
                sys.executable,
                str(mpnn_run),
                "--out_folder",
                str(out_folder),
                "--num_seq_per_target",
                str(num_seq_target),
                "--sampling_temp",
                str(temp),
                "--batch_size",
                str(num_seq_target),
                "--seed",
                seed,
            ]
            if weight_dir is not None:
                cmd.extend(["--path_to_model_weights", str(weight_dir)])
            else:
                if use_soluble_model:
                    cmd.append("--use_soluble_model")
                else:
                    raise StepError("Missing tools.mpnn_ckpt/abmpnn_ckpt for sequence design")
            if model_name:
                cmd.extend(["--model_name", model_name])
            elif protocol != "binder":
                cmd.extend(["--model_name", "abmpnn"])
            if use_soluble_model and protocol != "binder":
                cmd.append("--use_soluble_model")
            return cmd

        pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")
        if not run_stem:
            run_stem = pdb_path.stem
        if fixed_positions_jsonl is None and protocol in {"antibody", "vhh"} and step_name == "seq1":
            chains = [c.strip() for c in str(chain_list).split(",") if c.strip()]
            mapping = self._fixed_positions_from_bfactor(pdb_path, chains)
            fixed_positions_jsonl = self._write_fixed_positions_jsonl_from_mapping(
                item_tmp, run_stem, mapping
            )
        # Materialize legacy pdbs/ for downstream flowpacker (seq2).
        pdbs_dir = out_dir / "pdbs"
        pdbs_dir.mkdir(parents=True, exist_ok=True)
        legacy_pdb = pdbs_dir / f"{run_stem}.pdb"
        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        promote_file_atomic(pdb_path, legacy_pdb, allow_reuse=allow_reuse)
        tmp_pdb = item_tmp / f"{run_stem}.pdb"
        if not tmp_pdb.exists():
            try:
                os.link(pdb_path, tmp_pdb)
            except OSError:
                shutil.copy2(pdb_path, tmp_pdb)

        env = os.environ.copy()
        bias_enabled = bool(protocol == "binder" and self.cfg.get("name") == "seq1" and seq_cfg.get("bias_large_residues"))
        bias_num = int(seq_cfg.get("bias_num") or 0)
        if bias_enabled and bias_num > 0:
            vanilla_num = max(int(num_seq) - bias_num, 0)
            bias_residues = seq_cfg.get("bias_residues") or ["F", "M", "W"]
            if isinstance(bias_residues, str):
                bias_residues = [r.strip() for r in bias_residues.split(",") if r.strip()]
            bias_weight = float(seq_cfg.get("bias_weight") or 0.7)
            tmp_base = item_tmp / "_tmp_seq1"
            vanilla_dir = tmp_base / "vanilla"
            bias_dir = tmp_base / "bias"

            if vanilla_num > 0:
                base_cmd = build_base_cmd(vanilla_dir, vanilla_num, sampling_temp, use_soluble)
                cmd = base_cmd + ["--pdb_path", str(tmp_pdb)]
                if chain_arg:
                    cmd.extend(["--pdb_path_chains", chain_arg])
                if fixed_positions_jsonl:
                    cmd.extend(["--fixed_positions_jsonl", str(fixed_positions_jsonl)])
                if omit_aas:
                    cmd.extend(["--omit_AAs", str(omit_aas)])
                run_command(
                    cmd,
                    env=env,
                    cwd=str(item_tmp),
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )

            bias_jsonl = self._write_bias_jsonl(bias_dir, bias_residues, bias_weight)
            base_cmd = build_base_cmd(bias_dir, bias_num, sampling_temp, use_soluble)
            base_cmd.extend(["--bias_AA_jsonl", str(bias_jsonl)])
            cmd = base_cmd + ["--pdb_path", str(tmp_pdb)]
            if chain_arg:
                cmd.extend(["--pdb_path_chains", chain_arg])
            if fixed_positions_jsonl:
                cmd.extend(["--fixed_positions_jsonl", str(fixed_positions_jsonl)])
            run_command(
                cmd,
                env=env,
                cwd=str(item_tmp),
                log_file=self.cfg.get("_log_file"),
                verbose=bool(self.cfg.get("_verbose")),
            )

            self._merge_fastas(vanilla_dir, bias_dir, out_dir)
            shutil.rmtree(tmp_base, ignore_errors=True)
        else:
            base_cmd = build_base_cmd(item_tmp, num_seq, sampling_temp, use_soluble)
            cmd = base_cmd + ["--pdb_path", str(tmp_pdb)]
            if chain_arg:
                cmd.extend(["--pdb_path_chains", chain_arg])
            if fixed_positions_jsonl:
                cmd.extend(["--fixed_positions_jsonl", str(fixed_positions_jsonl)])
            if omit_aas:
                cmd.extend(["--omit_AAs", str(omit_aas)])
            run_command(
                cmd,
                env=env,
                cwd=str(item_tmp),
                log_file=self.cfg.get("_log_file"),
                verbose=bool(self.cfg.get("_verbose")),
            )

            seq_src = item_tmp / "seqs"
            if seq_src.exists():
                seq_dst = out_dir / "seqs"
                seq_dst.mkdir(parents=True, exist_ok=True)
                for fp in seq_src.glob("*.fa*"):
                    promote_file_atomic(fp, seq_dst / fp.name, allow_reuse=allow_reuse)
        shutil.rmtree(item_tmp, ignore_errors=True)

    def run_batch(self, ctx: StepContext, items: list[WorkItem]) -> dict[str, tuple[str, str | None]]:
        tools = ctx.input_data.get("tools") or {}
        out_dir = self.output_dir(ctx)
        batch_id = items[0].id if items else "batch"
        step_name = str(self.cfg.get("name") or self.name)
        if is_minimal(ctx):
            batch_dir = step_scratch_dir(ctx, step_name) / f"batch_{batch_id}"
        else:
            batch_dir = out_dir / ".tmp" / f"batch_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        pdb_dir = batch_dir / "pdbs"
        pdb_dir.mkdir(parents=True, exist_ok=True)

        mpnn_run = self._default_mpnn_run(ctx)
        if not mpnn_run or not mpnn_run.exists():
            raise StepError(
                f"MPNN runner not found. Set tools.mpnn_run/abmpnn_run or provide command in {self.cfg.get('config_path')}"
            )
        ckpt = tools.get("mpnn_ckpt") if ctx.input_data.get("protocol") == "binder" else tools.get("abmpnn_ckpt")

        seq_cfg = (ctx.input_data.get("sequence_design") or {}).get(
            "round1" if self.cfg.get("name") == "seq1" else "round2"
        ) or {}
        num_seq = int(seq_cfg.get("num_seq_per_backbone") or 0)
        sampling_temp = float(seq_cfg.get("sampling_temp") or 0.1)
        if num_seq <= 0:
            raise StepError("sequence_design.num_seq_per_backbone must be > 0")
        chain_list = self._default_chain_list(ctx)
        chain_arg = " ".join(str(chain_list).replace(",", " ").split())
        protocol = ctx.input_data.get("protocol")
        omit_aas = seq_cfg.get("omit_aas") if protocol != "binder" else None
        use_soluble = bool(seq_cfg.get("use_soluble_ckpt"))
        if protocol == "binder" and use_soluble and self.cfg.get("name") == "seq2":
            ckpt = tools.get("mpnn_ckpt_soluble") or ckpt

        weight_dir, model_name = self._resolve_weights(ckpt)
        seed = str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0))
        fixed_positions_csv = (items[0].payload or {}).get("fixed_positions_csv") if items else None
        default_chain = str((items[0].payload or {}).get("default_chain") or "A") if items else "A"

        results: dict[str, tuple[str, str | None]] = {}
        valid_items: list[WorkItem] = []
        staged_pdbs: list[Path] = []
        for item in items:
            pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
            if not pdb_path.exists():
                if self.item_done(ctx, item):
                    results[item.id] = ("done", None)
                else:
                    results[item.id] = ("blocked", f"PDB not found: {pdb_path}")
                continue
            run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
            # Materialize legacy pdbs/ for downstream flowpacker (seq2).
            pdbs_dir = out_dir / "pdbs"
            pdbs_dir.mkdir(parents=True, exist_ok=True)
            legacy_pdb = pdbs_dir / f"{run_stem}.pdb"
            allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
            promote_file_atomic(pdb_path, legacy_pdb, allow_reuse=allow_reuse)
            tmp_pdb = pdb_dir / f"{run_stem}.pdb"
            if not tmp_pdb.exists():
                try:
                    os.link(pdb_path, tmp_pdb)
                except OSError:
                    shutil.copy2(pdb_path, tmp_pdb)
            staged_pdbs.append(tmp_pdb)
            valid_items.append(item)

        if not valid_items:
            shutil.rmtree(batch_dir, ignore_errors=True)
            return results

        helper_dir = mpnn_run.parent / "helper_scripts"
        parse_script = helper_dir / "parse_multiple_chains.py"
        chain_script = helper_dir / "assign_fixed_chains.py"
        if not parse_script.exists() or not chain_script.exists():
            err = "ProteinMPNN helper_scripts missing (parse_multiple_chains.py/assign_fixed_chains.py)"
            for item in valid_items:
                if self.item_done(ctx, item):
                    results[item.id] = ("done", None)
                else:
                    results[item.id] = ("failed", err)
            shutil.rmtree(batch_dir, ignore_errors=True)
            return results

        parsed_jsonl = batch_dir / "parsed.jsonl"
        chain_jsonl = batch_dir / "chain_id.jsonl"
        fixed_positions_jsonl = None
        if fixed_positions_csv:
            fixed_positions_jsonl = self._write_fixed_positions_jsonl(
                batch_dir,
                staged_pdbs,
                str(fixed_positions_csv),
                default_chain,
            )
        if fixed_positions_jsonl is None and protocol in {"antibody", "vhh"} and step_name == "seq1":
            lines = []
            chains = [c.strip() for c in str(chain_list).split(",") if c.strip()]
            for pdb_path in staged_pdbs:
                mapping = self._fixed_positions_from_bfactor(pdb_path, chains)
                if not mapping:
                    continue
                lines.append(json.dumps({"name": pdb_path.stem, "fixed_positions": mapping}))
            if lines:
                fixed_positions_jsonl = batch_dir / "fixed_positions.jsonl"
                fixed_positions_jsonl.write_text("\n".join(lines) + "\n")

        def build_base_cmd(out_folder: Path, num_seq_target: int, temp: float, use_soluble_model: bool) -> list[str]:
            cmd = [
                sys.executable,
                str(mpnn_run),
                "--out_folder",
                str(out_folder),
                "--num_seq_per_target",
                str(num_seq_target),
                "--sampling_temp",
                str(temp),
                "--batch_size",
                str(num_seq_target),
                "--seed",
                seed,
                "--jsonl_path",
                str(parsed_jsonl),
                "--chain_id_jsonl",
                str(chain_jsonl),
            ]
            if fixed_positions_jsonl:
                cmd.extend(["--fixed_positions_jsonl", str(fixed_positions_jsonl)])
            if weight_dir is not None:
                cmd.extend(["--path_to_model_weights", str(weight_dir)])
            else:
                if use_soluble_model:
                    cmd.append("--use_soluble_model")
                else:
                    raise StepError("Missing tools.mpnn_ckpt/abmpnn_ckpt for sequence design")
            if model_name:
                cmd.extend(["--model_name", model_name])
            elif protocol != "binder":
                cmd.extend(["--model_name", "abmpnn"])
            if use_soluble_model and protocol != "binder":
                cmd.append("--use_soluble_model")
            if omit_aas:
                cmd.extend(["--omit_AAs", str(omit_aas)])
            return cmd

        env = os.environ.copy()
        bias_enabled = bool(protocol == "binder" and self.cfg.get("name") == "seq1" and seq_cfg.get("bias_large_residues"))
        bias_num = int(seq_cfg.get("bias_num") or 0)
        err: str | None = None
        try:
            # Build parsed jsonl for batch inputs.
            run_command(
                [
                    sys.executable,
                    str(parse_script),
                    "--input_path",
                    str(pdb_dir),
                    "--output_path",
                    str(parsed_jsonl),
                ],
                env=os.environ.copy(),
                cwd=str(batch_dir),
                log_file=self.cfg.get("_log_file"),
                verbose=bool(self.cfg.get("_verbose")),
            )
            # Build chain assignment jsonl.
            run_command(
                [
                    sys.executable,
                    str(chain_script),
                    "--input_path",
                    str(parsed_jsonl),
                    "--output_path",
                    str(chain_jsonl),
                    "--chain_list",
                    chain_arg,
                ],
                env=os.environ.copy(),
                cwd=str(batch_dir),
                log_file=self.cfg.get("_log_file"),
                verbose=bool(self.cfg.get("_verbose")),
            )

            if bias_enabled and bias_num > 0:
                vanilla_num = max(int(num_seq) - bias_num, 0)
                bias_residues = seq_cfg.get("bias_residues") or ["F", "M", "W"]
                if isinstance(bias_residues, str):
                    bias_residues = [r.strip() for r in bias_residues.split(",") if r.strip()]
                bias_weight = float(seq_cfg.get("bias_weight") or 0.7)
                tmp_base = batch_dir / "_tmp_seq1"
                vanilla_dir = tmp_base / "vanilla"
                bias_dir = tmp_base / "bias"

                if vanilla_num > 0:
                    base_cmd = build_base_cmd(vanilla_dir, vanilla_num, sampling_temp, use_soluble)
                    run_command(
                        base_cmd,
                        env=env,
                        cwd=str(batch_dir),
                        log_file=self.cfg.get("_log_file"),
                        verbose=bool(self.cfg.get("_verbose")),
                    )

                bias_jsonl = self._write_bias_jsonl(bias_dir, bias_residues, bias_weight)
                base_cmd = build_base_cmd(bias_dir, bias_num, sampling_temp, use_soluble)
                base_cmd.extend(["--bias_AA_jsonl", str(bias_jsonl)])
                run_command(
                    base_cmd,
                    env=env,
                    cwd=str(batch_dir),
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )

                self._merge_fastas(vanilla_dir, bias_dir, out_dir)
                shutil.rmtree(tmp_base, ignore_errors=True)
            else:
                base_cmd = build_base_cmd(batch_dir, num_seq, sampling_temp, use_soluble)
                run_command(
                    base_cmd,
                    env=env,
                    cwd=str(batch_dir),
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )

                seq_src = batch_dir / "seqs"
                if seq_src.exists():
                    seq_dst = out_dir / "seqs"
                    seq_dst.mkdir(parents=True, exist_ok=True)
                    for fp in seq_src.glob("*.fa*"):
                        promote_file_atomic(
                            fp,
                            seq_dst / fp.name,
                            allow_reuse=bool((ctx.work_queue or {}).get("allow_reuse", True)),
                        )
        except Exception as exc:
            err = str(exc)
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

        for item in valid_items:
            if self.item_done(ctx, item):
                results[item.id] = ("done", None)
            else:
                results[item.id] = ("failed", err or "missing output")
        return results

    def run_full(self, ctx: StepContext) -> None:
        if self.cfg.get("command"):
            return super().run_full(ctx)

        tools = ctx.input_data.get("tools") or {}
        input_dir = self.cfg.get("input_dir")
        if not input_dir:
            raise StepError(f"Step {self.name} missing input_dir")
        input_dir = str((ctx.out_dir / input_dir) if not Path(input_dir).is_absolute() else input_dir)
        if not Path(input_dir).exists():
            raise StepError(f"Input dir not found: {input_dir}")
        out_dir = str(self.output_dir(ctx))

        mpnn_run = self._default_mpnn_run(ctx)
        if not mpnn_run or not mpnn_run.exists():
            raise StepError(
                f"MPNN runner not found. Set tools.mpnn_run/abmpnn_run or provide command in {self.cfg.get('config_path')}"
            )
        ckpt = tools.get("mpnn_ckpt") if ctx.input_data.get("protocol") == "binder" else tools.get("abmpnn_ckpt")

        seq_cfg = (ctx.input_data.get("sequence_design") or {}).get("round1" if self.cfg.get("name") == "seq1" else "round2") or {}
        num_seq = int(seq_cfg.get("num_seq_per_backbone") or 0)
        sampling_temp = float(seq_cfg.get("sampling_temp") or 0.1)
        if num_seq <= 0:
            raise StepError("sequence_design.num_seq_per_backbone must be > 0")
        chain_list = self._default_chain_list(ctx)
        position_list = self.cfg.get("fixed_positions_csv")
        protocol = ctx.input_data.get("protocol")
        omit_aas = seq_cfg.get("omit_aas") if protocol != "binder" else None
        use_soluble = bool(seq_cfg.get("use_soluble_ckpt"))
        if protocol == "binder" and use_soluble and self.cfg.get("name") == "seq2":
            ckpt = tools.get("mpnn_ckpt_soluble") or ckpt

        weight_dir, model_name = self._resolve_weights(ckpt)
        chain_arg = " ".join(str(chain_list).replace(",", " ").split())
        seed = str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0))

        def build_base_cmd(out_folder: Path, num_seq_target: int, temp: float, use_soluble_model: bool) -> list[str]:
            cmd = [
                sys.executable,
                str(mpnn_run),
                "--out_folder",
                str(out_folder),
                "--num_seq_per_target",
                str(num_seq_target),
                "--sampling_temp",
                str(temp),
                "--batch_size",
                str(num_seq_target),
                "--seed",
                seed,
            ]
            if weight_dir is not None:
                cmd.extend(["--path_to_model_weights", str(weight_dir)])
            else:
                if use_soluble_model:
                    cmd.append("--use_soluble_model")
                else:
                    raise StepError("Missing tools.mpnn_ckpt/abmpnn_ckpt for sequence design")
            if model_name:
                cmd.extend(["--model_name", model_name])
            elif protocol != "binder":
                cmd.extend(["--model_name", "abmpnn"])
            if use_soluble_model and protocol != "binder":
                cmd.append("--use_soluble_model")
            if omit_aas:
                cmd.extend(["--omit_AAs", str(omit_aas)])
            return cmd


        step_label = str(self.cfg.get("name") or self.name)

        pdbs = collect_pdbs(Path(input_dir))
        if not pdbs:
            raise StepError(f"No PDBs found for sequence design in {input_dir}")

        # If PDB basenames collide (e.g., partial flow sample0.pdb in multiple subdirs),
        # stage unique names and use those for MPNN to avoid seq output collisions.
        staged_pdbs = pdbs
        stems = [p.stem for p in pdbs]
        should_stage = self.cfg.get("name") == "seq2" or len(set(stems)) != len(stems)
        if should_stage:
            stage_dir = Path(out_dir) / "pdbs"
            stage_dir.mkdir(parents=True, exist_ok=True)
            staged_pdbs = []
            for pdb_path in pdbs:
                parent = pdb_path.parent.name
                unique_stem = f"{parent}__{pdb_path.stem}" if should_stage else pdb_path.stem
                dst = stage_dir / f"{unique_stem}.pdb"
                if not dst.exists():
                    try:
                        os.link(pdb_path, dst)
                    except OSError:
                        shutil.copy2(pdb_path, dst)
                staged_pdbs.append(dst)
            pdbs = staged_pdbs

        fixed_positions_jsonl = None
        if position_list:
            fixed_positions_jsonl = self._write_fixed_positions_jsonl(
                Path(out_dir),
                pdbs,
                position_list,
                str(chain_list).split(",")[0].strip() or "A",
            )
            if not fixed_positions_jsonl:
                print("Warning: fixed_positions_csv provided but could not generate fixed_positions_jsonl; skipping anchors.")
        if fixed_positions_jsonl is None and protocol in {"antibody", "vhh"} and step_label == "seq1":
            lines = []
            chains = [c.strip() for c in str(chain_list).split(",") if c.strip()]
            for pdb_path in pdbs:
                mapping = self._fixed_positions_from_bfactor(pdb_path, chains)
                if not mapping:
                    continue
                lines.append(json.dumps({"name": pdb_path.stem, "fixed_positions": mapping}))
            if lines:
                fixed_positions_jsonl = Path(out_dir) / "fixed_positions.jsonl"
                fixed_positions_jsonl.write_text("\n".join(lines) + "\n")

        env = os.environ.copy()

        bias_enabled = bool(protocol == "binder" and self.cfg.get("name") == "seq1" and seq_cfg.get("bias_large_residues"))
        bias_num = int(seq_cfg.get("bias_num") or 0)
        if bias_enabled and bias_num > 0:
            vanilla_num = max(int(num_seq) - bias_num, 0)
            bias_residues = seq_cfg.get("bias_residues") or ["F", "M", "W"]
            if isinstance(bias_residues, str):
                bias_residues = [r.strip() for r in bias_residues.split(",") if r.strip()]
            bias_weight = float(seq_cfg.get("bias_weight") or 0.7)
            tmp_base = self.output_dir(ctx) / "_tmp_seq1"
            vanilla_dir = tmp_base / "vanilla"
            bias_dir = tmp_base / "bias"

            if vanilla_num > 0:
                base_cmd = build_base_cmd(vanilla_dir, vanilla_num, sampling_temp, use_soluble)
                total = len(pdbs)
                for idx, pdb_path in enumerate(pdbs, start=1):
                    cmd = base_cmd + ["--pdb_path", str(pdb_path)]
                    if chain_arg:
                        cmd.extend(["--pdb_path_chains", chain_arg])
                    if fixed_positions_jsonl:
                        cmd.extend(["--fixed_positions_jsonl", str(fixed_positions_jsonl)])
                    start = time.time()
                    status = "OK"
                    try:
                        run_command(
                            cmd,
                            env=env,
                            log_file=self.cfg.get("_log_file"),
                            verbose=bool(self.cfg.get("_verbose")),
                        )
                    except Exception:
                        status = "FAILED"
                        raise
                    finally:
                        log_command_progress(
                            step_label,
                            idx,
                            total,
                            item=pdb_path.stem,
                            phase="vanilla",
                            status=status,
                            elapsed=time.time() - start,
                            log_file=self.cfg.get("_log_file"),
                        )

            bias_jsonl = self._write_bias_jsonl(bias_dir, bias_residues, bias_weight)
            base_cmd = build_base_cmd(bias_dir, bias_num, sampling_temp, use_soluble)
            base_cmd.extend(["--bias_AA_jsonl", str(bias_jsonl)])
            total = len(pdbs)
            for idx, pdb_path in enumerate(pdbs, start=1):
                cmd = base_cmd + ["--pdb_path", str(pdb_path)]
                if chain_arg:
                    cmd.extend(["--pdb_path_chains", chain_arg])
                if fixed_positions_jsonl:
                    cmd.extend(["--fixed_positions_jsonl", str(fixed_positions_jsonl)])
                start = time.time()
                status = "OK"
                try:
                    run_command(
                        cmd,
                        env=env,
                        log_file=self.cfg.get("_log_file"),
                        verbose=bool(self.cfg.get("_verbose")),
                    )
                except Exception:
                    status = "FAILED"
                    raise
                finally:
                    log_command_progress(
                        step_label,
                        idx,
                        total,
                        item=pdb_path.stem,
                        phase="bias",
                        status=status,
                        elapsed=time.time() - start,
                        log_file=self.cfg.get("_log_file"),
                    )

            self._merge_fastas(vanilla_dir, bias_dir, Path(out_dir))
            shutil.rmtree(tmp_base, ignore_errors=True)
        else:
            base_cmd = build_base_cmd(Path(out_dir), num_seq, sampling_temp, use_soluble)
            total = len(pdbs)
            for idx, pdb_path in enumerate(pdbs, start=1):
                cmd = base_cmd + ["--pdb_path", str(pdb_path)]
                if chain_arg:
                    cmd.extend(["--pdb_path_chains", chain_arg])
                if fixed_positions_jsonl:
                    cmd.extend(["--fixed_positions_jsonl", str(fixed_positions_jsonl)])
                start = time.time()
                status = "OK"
                try:
                    run_command(
                        cmd,
                        env=env,
                        log_file=self.cfg.get("_log_file"),
                        verbose=bool(self.cfg.get("_verbose")),
                    )
                except Exception:
                    status = "FAILED"
                    raise
                finally:
                    log_command_progress(
                        step_label,
                        idx,
                        total,
                        item=pdb_path.stem,
                        status=status,
                        elapsed=time.time() - start,
                        log_file=self.cfg.get("_log_file"),
                    )

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        rows = []
        for fp in sorted(out_dir.rglob("*.fa")) + sorted(out_dir.rglob("*.fasta")):
            rows.append({
                "design_id": extract_design_id(fp.stem),
                "structure_id": structure_id_from_name(fp.stem),
                "fasta_path": str(fp),
            })
        if not rows:
            return
        write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "fasta_path"])

    def scan_done(self, ctx: StepContext) -> set[int]:
        done = super().scan_done(ctx)
        if done:
            return done
        out_dir = self.output_dir(ctx)
        step_name = str(self.cfg.get("name") or self.name)
        if step_name == "seq2":
            pdb_dir = out_dir / "pdbs"
            if not pdb_dir.exists() or not list(pdb_dir.glob("*.pdb")):
                return set()
        seq_dir = out_dir / "seqs"
        fasta_files = list(seq_dir.glob("*.fa*")) if seq_dir.exists() else []
        if not fasta_files:
            fasta_files = list(out_dir.glob("*.fa*"))
        if fasta_files:
            return set(range(self.expected_total(ctx)))
        return set()


class FlowPackerStep(ExternalCommandStep):
    name = "flowpacker"
    stage = "score"
    supports_work_queue = True
    work_queue_mode = "items"
    # Drain per-worker batches to avoid repeated model reloads.
    per_worker_batch = True
    # 0 means "claim all available items for this worker".
    batch_size = 0

    def _flowpacker_config(self) -> dict:
        cfg = dict(self.cfg.get("flowpacker") or {})
        if cfg:
            return cfg
        cmd = self.cfg.get("command") or []
        if isinstance(cmd, str):
            cmd = [cmd]
        out: dict[str, object] = {}
        for idx, token in enumerate(cmd):
            if token in {
                "--input_pdb_dir",
                "--seq_fasta_dir",
                "--output_dir",
                "--flowpacker_repo",
                "--base_yaml",
                "--num_jobs",
                "--binder_chain",
                "--link_suffix",
            }:
                if idx + 1 < len(cmd):
                    out[token.lstrip("-")] = cmd[idx + 1]
        return out

    def _resolve_path(self, ctx: StepContext, value: str | None) -> Path | None:
        if not value:
            return None
        p = Path(str(value))
        if p.is_absolute():
            return p
        return ctx.out_dir / p

    def _read_fasta(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        text = ""
        if path.suffix == ".gz":
            import gzip

            with gzip.open(path, "rt") as handle:
                text = handle.read()
        else:
            text = path.read_text()
        seqs: list[str] = []
        current: list[str] = []
        started = False
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if started:
                    seqs.append("".join(current))
                current = []
                started = True
            else:
                current.append(line)
        if started:
            seqs.append("".join(current))
        return seqs

    def _find_fasta(self, seq_dir: Path, stem: str) -> Path | None:
        for suffix in [".fa", ".fasta", ".fa.gz", ".fasta.gz"]:
            cand = seq_dir / f"{stem}{suffix}"
            if cand.exists():
                return cand
        return None

    def _write_seq_csv(self, fasta_path: Path, csv_path: Path, link_name: str) -> int:
        seqs = self._read_fasta(fasta_path)
        if not seqs:
            return 0
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        seen = set()
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["link_name", "seq", "seq_idx"])
            for i, seq in enumerate(seqs):
                if i == 0:
                    continue
                if not seq or seq in seen:
                    continue
                seen.add(seq)
                writer.writerow([link_name, seq, str(i)])
                count += 1
        return count

    def _resolve_base_yaml(self, flowpacker_repo: Path, base_yaml: str | None) -> Path:
        if base_yaml:
            return Path(base_yaml)
        candidates = [
            flowpacker_repo / "config" / "inference" / "base.yaml",
            flowpacker_repo / "config" / "inference" / "base.yml",
            flowpacker_repo / "config" / "base.yaml",
            flowpacker_repo / "config" / "base.yml",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError("Could not locate FlowPacker base.yaml; pass --base_yaml explicitly.")

    def _abspath_if_relative(self, value: str | None, root: Path) -> str | None:
        if not value:
            return value
        p = Path(str(value))
        if p.is_absolute():
            return str(p)
        return str((root / p).resolve())

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        return self.list_items(ctx, readonly=False)

    def list_items(self, ctx: StepContext, *, readonly: bool = False) -> list[WorkItem]:
        cfg = self._flowpacker_config()
        input_pdb_dir = self._resolve_path(ctx, cfg.get("input_pdb_dir")) if cfg else None
        seq_fasta_dir = self._resolve_path(ctx, cfg.get("seq_fasta_dir")) if cfg else None
        if not input_pdb_dir or not input_pdb_dir.exists():
            step_name = str(self.cfg.get("name") or self.name)
            if step_name == "flowpacker2":
                alt_root = ctx.out_dir / "output" / "partial_flow"
                if alt_root.exists():
                    alt_pdbs = collect_pdbs(alt_root)
                    if alt_pdbs:
                        if readonly:
                            input_pdb_dir = alt_root
                        else:
                            stage_dir = ctx.out_dir / "output" / "seqs_round2" / "pdbs"
                            stage_dir.mkdir(parents=True, exist_ok=True)
                            run_stems = compute_run_stems(alt_pdbs, alt_root)
                            for pdb_path in alt_pdbs:
                                run_stem = run_stems[pdb_path]
                                dst = stage_dir / f"{run_stem}.pdb"
                                if not dst.exists():
                                    try:
                                        os.link(pdb_path, dst)
                                    except OSError:
                                        shutil.copy2(pdb_path, dst)
                            input_pdb_dir = stage_dir
            if not input_pdb_dir or not input_pdb_dir.exists():
                raise StepError("flowpacker input_pdb_dir missing or invalid")
        if not seq_fasta_dir or not seq_fasta_dir.exists():
            raise StepError("flowpacker seq_fasta_dir missing or invalid")

        pdbs = collect_pdbs(input_pdb_dir)
        if not pdbs:
            raise StepError(f"No PDBs found for flowpacker in {input_pdb_dir}")
        run_stems = compute_run_stems(pdbs, input_pdb_dir)

        items: list[WorkItem] = []
        for pdb_path in pdbs:
            fasta_path = self._find_fasta(seq_fasta_dir, pdb_path.stem)
            run_stem = run_stems[pdb_path]
            items.append(
                WorkItem(
                    id=run_stem,
                    payload={
                        "pdb_path": str(pdb_path),
                        "pdb_stem": pdb_path.stem,
                        "run_stem": run_stem,
                        "fasta_path": str(fasta_path) if fasta_path else None,
                    },
                )
            )
        return items

    def item_done(self, ctx: StepContext, item: WorkItem) -> bool:
        out_dir = self.output_dir(ctx)
        packed_dir = out_dir / "packed_pdbs"
        if not packed_dir.exists():
            return False
        stem = str((item.payload or {}).get("run_stem") or (item.payload or {}).get("pdb_stem") or item.id)
        cfg = self._flowpacker_config()
        link_suffix = str(cfg.get("link_suffix") or ".pdb")
        link_name = f"{stem}{link_suffix}"
        if ".pdb" in link_name:
            pattern = link_name.replace(".pdb", "_*.pdb")
        else:
            pattern = f"{link_name}_*"
        return any(packed_dir.glob(pattern))

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        cfg = self._flowpacker_config()
        flowpacker_repo = self._resolve_path(ctx, cfg.get("flowpacker_repo")) if cfg else None
        if not flowpacker_repo or not flowpacker_repo.exists():
            raise StepError("flowpacker_repo missing or invalid")

        pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        fasta_path = (item.payload or {}).get("fasta_path")
        if not fasta_path:
            raise FileNotFoundError(f"Missing fasta for {pdb_path.stem}")
        fasta_path = Path(str(fasta_path))
        if not fasta_path.exists():
            raise FileNotFoundError(f"Fasta not found: {fasta_path}")

        binder_chain = str(cfg.get("binder_chain") or (ctx.input_data.get("binder_chain") or "A"))
        link_suffix = str(cfg.get("link_suffix") or ".pdb")
        base_yaml = cfg.get("base_yaml")

        out_dir = self.output_dir(ctx)
        step_name = str(self.cfg.get("name") or self.name)
        if is_minimal(ctx):
            item_dir = step_scratch_dir(ctx, step_name) / item.id
        else:
            item_dir = out_dir / ".tmp" / item.id
        item_dir.mkdir(parents=True, exist_ok=True)
        pdb_dir = item_dir / "input_pdb"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
        dst = pdb_dir / f"{run_stem}.pdb"
        if not dst.exists():
            try:
                os.link(pdb_path, dst)
            except OSError:
                shutil.copy2(pdb_path, dst)

        csv_path = item_dir / "flowpacker_input.csv"
        link_name = f"{run_stem}{link_suffix}"
        seq_count = self._write_seq_csv(fasta_path, csv_path, link_name)
        if seq_count == 0:
            raise StepError(f"No sequences written for {pdb_path.stem}")

        yaml_path = item_dir / "input.yaml"
        base_yaml_path = self._resolve_base_yaml(flowpacker_repo, base_yaml)
        try:
            import yaml
        except Exception as exc:
            raise StepError(f"Missing yaml dependency: {exc}") from exc
        base_cfg = yaml.safe_load(base_yaml_path.read_text())
        if isinstance(base_cfg, dict):
            base_cfg["ckpt"] = self._abspath_if_relative(base_cfg.get("ckpt"), flowpacker_repo)
            base_cfg["conf_ckpt"] = self._abspath_if_relative(base_cfg.get("conf_ckpt"), flowpacker_repo)
        cfg_payload = dict(base_cfg or {})
        cfg_payload.setdefault("data", {})
        cfg_payload["data"]["test_path"] = str(pdb_dir)
        yaml_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))

        item_out = item_dir
        script = Path(__file__).resolve().parents[3] / "scripts" / "run_flowpacker.py"
        cmd = [
            sys.executable,
            str(script),
            "--batch_yaml",
            str(yaml_path),
            "--csv_file",
            str(csv_path),
            "--output_dir",
            str(item_out),
            "--flowpacker_repo",
            str(flowpacker_repo),
            "--binder_chain",
            str(binder_chain),
        ]
        run_command(
            cmd,
            env=os.environ.copy(),
            cwd=str(item_dir),
            log_file=self.cfg.get("_log_file"),
            verbose=bool(self.cfg.get("_verbose")),
        )

        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        packed_pdbs = out_dir / "packed_pdbs"
        promote_tree(item_out / "flowpacker_outputs" / "run_1", packed_pdbs, allow_reuse=allow_reuse)
        if not is_minimal(ctx):
            after_pdbs = out_dir / "after_pdbs"
            promote_tree(item_out / "after_pdbs", after_pdbs, allow_reuse=allow_reuse)
        shutil.rmtree(item_dir, ignore_errors=True)

    def run_batch(self, ctx: StepContext, items: list[WorkItem]) -> dict[str, tuple[str, str | None]]:
        cfg = self._flowpacker_config()
        flowpacker_repo = self._resolve_path(ctx, cfg.get("flowpacker_repo")) if cfg else None
        if not flowpacker_repo or not flowpacker_repo.exists():
            raise StepError("flowpacker_repo missing or invalid")

        binder_chain = str(cfg.get("binder_chain") or (ctx.input_data.get("binder_chain") or "A"))
        link_suffix = str(cfg.get("link_suffix") or ".pdb")
        base_yaml = cfg.get("base_yaml")

        out_dir = self.output_dir(ctx)
        batch_id = items[0].id if items else "batch"
        step_name = str(self.cfg.get("name") or self.name)
        if is_minimal(ctx):
            batch_dir = step_scratch_dir(ctx, step_name) / f"batch_{batch_id}"
        else:
            batch_dir = out_dir / ".tmp" / f"batch_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        pdb_dir = batch_dir / "input_pdb"
        pdb_dir.mkdir(parents=True, exist_ok=True)

        csv_path = batch_dir / "flowpacker_input.csv"
        results: dict[str, tuple[str, str | None]] = {}
        seq_count = 0
        seen: dict[str, set[str]] = {}
        valid_items: list[WorkItem] = []
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["link_name", "seq", "seq_idx"])
            for item in items:
                pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
                if not pdb_path.exists():
                    if self.item_done(ctx, item):
                        results[item.id] = ("done", None)
                    else:
                        results[item.id] = ("blocked", f"PDB not found: {pdb_path}")
                    continue
                fasta_path = (item.payload or {}).get("fasta_path")
                if not fasta_path:
                    if self.item_done(ctx, item):
                        results[item.id] = ("done", None)
                    else:
                        results[item.id] = ("blocked", f"Missing fasta for {pdb_path.stem}")
                    continue
                fasta_path = Path(str(fasta_path))
                if not fasta_path.exists():
                    if self.item_done(ctx, item):
                        results[item.id] = ("done", None)
                    else:
                        results[item.id] = ("blocked", f"Fasta not found: {fasta_path}")
                    continue
                run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
                dst = pdb_dir / f"{run_stem}.pdb"
                if not dst.exists():
                    try:
                        os.link(pdb_path, dst)
                    except OSError:
                        shutil.copy2(pdb_path, dst)
                link_name = f"{run_stem}{link_suffix}"
                seen_for_link = seen.setdefault(link_name, set())
                seqs = self._read_fasta(fasta_path)
                for i, seq in enumerate(seqs):
                    if i == 0 or not seq or seq in seen_for_link:
                        continue
                    seen_for_link.add(seq)
                    writer.writerow([link_name, seq, str(i)])
                    seq_count += 1
                valid_items.append(item)
        if not valid_items:
            shutil.rmtree(batch_dir, ignore_errors=True)
            return results
        if seq_count == 0:
            err = "No sequences written for flowpacker batch"
            for item in valid_items:
                results[item.id] = ("failed", err)
            shutil.rmtree(batch_dir, ignore_errors=True)
            return results

        yaml_path = batch_dir / "input.yaml"
        base_yaml_path = self._resolve_base_yaml(flowpacker_repo, base_yaml)
        try:
            import yaml
        except Exception as exc:
            raise StepError(f"Missing yaml dependency: {exc}") from exc
        base_cfg = yaml.safe_load(base_yaml_path.read_text())
        if isinstance(base_cfg, dict):
            base_cfg["ckpt"] = self._abspath_if_relative(base_cfg.get("ckpt"), flowpacker_repo)
            base_cfg["conf_ckpt"] = self._abspath_if_relative(base_cfg.get("conf_ckpt"), flowpacker_repo)
        cfg_payload = dict(base_cfg or {})
        cfg_payload.setdefault("data", {})
        cfg_payload["data"]["test_path"] = str(pdb_dir)
        yaml_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))

        script = Path(__file__).resolve().parents[3] / "scripts" / "run_flowpacker.py"
        cmd = [
            sys.executable,
            str(script),
            "--batch_yaml",
            str(yaml_path),
            "--csv_file",
            str(csv_path),
            "--output_dir",
            str(batch_dir),
            "--flowpacker_repo",
            str(flowpacker_repo),
            "--binder_chain",
            str(binder_chain),
        ]
        err: str | None = None
        if valid_items:
            try:
                run_command(
                    cmd,
                    env=os.environ.copy(),
                    cwd=str(batch_dir),
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )
            except Exception as exc:
                err = str(exc)

        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        strict_collision = False
        packed_pdbs = out_dir / "packed_pdbs"
        try:
            promote_tree(batch_dir / "flowpacker_outputs" / "run_1", packed_pdbs, allow_reuse=allow_reuse)
            if not is_minimal(ctx):
                after_pdbs = out_dir / "after_pdbs"
                promote_tree(batch_dir / "after_pdbs", after_pdbs, allow_reuse=allow_reuse)
        except Exception as exc:
            if err is None:
                err = str(exc)
            if not allow_reuse:
                strict_collision = True
        shutil.rmtree(batch_dir, ignore_errors=True)

        for item in valid_items:
            if strict_collision:
                results[item.id] = ("failed", err or "collision")
            elif self.item_done(ctx, item):
                results[item.id] = ("done", None)
            else:
                results[item.id] = ("failed", err or "missing output")
        return results

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        pdb_root = out_dir / "packed_pdbs"
        # Write legacy flowpacker_input.csv if possible.
        cfg = self._flowpacker_config()
        seq_fasta_dir = self._resolve_path(ctx, cfg.get("seq_fasta_dir")) if cfg else None
        link_suffix = str(cfg.get("link_suffix") or ".pdb")
        if seq_fasta_dir and seq_fasta_dir.exists():
            csv_path = out_dir / "flowpacker_input.csv"
            if not csv_path.exists():
                seq_count = 0
                seen = set()
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["link_name", "seq", "seq_idx"])
                    for fasta in sorted(seq_fasta_dir.rglob("*.fa*")):
                        if is_ignored_path(fasta):
                            continue
                        base_name = fasta.stem
                        seqs = self._read_fasta(fasta)
                        for i, seq in enumerate(seqs):
                            if i == 0 or not seq or seq in seen:
                                continue
                            seen.add(seq)
                            link_name = f"{base_name}{link_suffix}"
                            writer.writerow([link_name, seq, str(i)])
                            seq_count += 1
                if seq_count == 0:
                    try:
                        csv_path.unlink()
                    except Exception:
                        pass

        rows = []
        if not pdb_root.exists():
            return
        for fp in sorted(pdb_root.rglob("*.pdb")):
            if is_ignored_path(fp):
                continue
            rows.append({
                "design_id": extract_design_id(fp.stem),
                "structure_id": structure_id_from_name(fp.stem),
                "pdb_path": str(fp),
            })
        if not rows:
            return
        write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "pdb_path"])

    def scan_done(self, ctx: StepContext) -> set[int]:
        done = super().scan_done(ctx)
        if done:
            return done
        out_dir = self.output_dir(ctx)
        packed_dir = out_dir / "packed_pdbs"
        if packed_dir.exists():
            for fp in packed_dir.rglob("*.pdb"):
                if is_ignored_path(fp):
                    continue
                return set(range(self.expected_total(ctx)))
        return set()


class AF3ScoreStep(ExternalCommandStep):
    name = "af3score"
    stage = "score"
    supports_work_queue = True
    work_queue_mode = "items"
    # Drain per-worker batches to avoid repeated model reloads.
    per_worker_batch = True
    # 0 means "claim all available items for this worker".
    batch_size = 0

    def _af3score_config(self) -> dict:
        cfg = dict(self.cfg.get("af3score") or {})
        if cfg:
            return cfg
        cmd = self.cfg.get("command") or []
        if isinstance(cmd, str):
            cmd = [cmd]
        out: dict[str, object] = {}
        for idx, token in enumerate(cmd):
            if token in {
                "--input_pdb_dir",
                "--output_dir",
                "--af3score_repo",
                "--model_dir",
                "--db_dir",
                "--num_jobs",
                "--num_workers",
                "--bucket_default",
                "--num_samples",
                "--model_seeds",
                "--write_cif_model",
                "--export_pdb_dir",
                "--export_cif_dir",
                "--target_offsets_json",
                "--target_chain",
            }:
                if idx + 1 < len(cmd):
                    out[token.lstrip("-")] = cmd[idx + 1]
            elif token == "--no_templates":
                out["no_templates"] = True
            elif token == "--write_best_model_root":
                out["write_best_model_root"] = True
            elif token == "--write_ranking_scores_csv":
                out["write_ranking_scores_csv"] = True
        return out

    def _resolve_path(self, ctx: StepContext, value: str | None) -> Path | None:
        if not value:
            return None
        p = Path(str(value))
        if p.is_absolute():
            return p
        return ctx.out_dir / p

    def _resolve_input_dir(self, input_dir: Path) -> Path:
        if (input_dir / "run_1").exists():
            return input_dir / "run_1"
        return input_dir

    def _merge_tree(self, src: Path, dst: Path) -> None:
        if not src.exists():
            return
        for root, _, files in os.walk(src):
            root_path = Path(root)
            rel = root_path.relative_to(src)
            dest_dir = dst / rel
            dest_dir.mkdir(parents=True, exist_ok=True)
            for fname in files:
                src_file = root_path / fname
                dst_file = dest_dir / fname
                if dst_file.exists():
                    continue
                try:
                    os.link(src_file, dst_file)
                except Exception:
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception:
                        pass

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        return self.list_items(ctx, readonly=False)

    def list_items(self, ctx: StepContext, *, readonly: bool = False) -> list[WorkItem]:
        _ = readonly  # list_items must be read-only for rebuild/output checks.
        cfg = self._af3score_config()
        input_dir = self.cfg.get("input_dir") or (cfg.get("input_pdb_dir") if cfg else None)
        if not input_dir:
            raise StepError("af3score input_dir missing or invalid")
        input_pdb_dir = Path(str(input_dir))
        if not input_pdb_dir.is_absolute():
            input_pdb_dir = ctx.out_dir / input_pdb_dir
        if not input_pdb_dir.exists():
            raise StepError(f"af3score input_dir missing or invalid: {input_pdb_dir}")
        input_pdb_dir = self._resolve_input_dir(input_pdb_dir)

        pdbs = collect_pdbs(input_pdb_dir)
        if not pdbs:
            raise StepError(f"No PDBs found for af3score in {input_pdb_dir}")
        run_stems = compute_run_stems(pdbs, input_pdb_dir)

        items: list[WorkItem] = []
        for pdb_path in pdbs:
            run_stem = run_stems[pdb_path]
            items.append(
                WorkItem(
                    id=run_stem,
                    payload={
                        "pdb_path": str(pdb_path),
                        "pdb_name": pdb_path.stem,
                        "run_stem": run_stem,
                    },
                )
            )
        return items

    def item_done(self, ctx: StepContext, item: WorkItem) -> bool:
        out_dir = self.output_dir(ctx)
        metrics_path = out_dir / "metrics_items" / f"{item.id}.csv"
        cif_path = out_dir / "cif" / f"{item.id}.cif"
        return is_valid_metrics_shard(metrics_path, expected_desc=str(item.id)) and cif_path.exists()

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        cfg = self._af3score_config()
        af3_repo = self._resolve_path(ctx, cfg.get("af3score_repo")) if cfg else None
        model_dir = self._resolve_path(ctx, cfg.get("model_dir")) if cfg else None
        if not af3_repo or not af3_repo.exists():
            raise StepError("af3score_repo missing or invalid")
        if not model_dir or not model_dir.exists():
            raise StepError("model_dir missing or invalid")

        pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
        try:
            attempt = int((item.payload or {}).get("_attempt") or 1)
        except Exception:
            attempt = 1

        step_name = str(self.cfg.get("name") or self.name)
        scratch_root = step_scratch_dir(ctx, step_name) / f"attempt_{attempt}" / "items" / item.id
        scratch_root.mkdir(parents=True, exist_ok=True)
        tmp_pdb = scratch_root / f"{run_stem}.pdb"
        if not tmp_pdb.exists():
            try:
                os.link(pdb_path, tmp_pdb)
            except OSError:
                shutil.copy2(pdb_path, tmp_pdb)

        script = None
        cmd_cfg = self.cfg.get("command") or []
        if isinstance(cmd_cfg, str):
            cmd_cfg = [cmd_cfg]
        for tok in cmd_cfg:
            if str(tok).endswith("run_af3score.py") and Path(str(tok)).exists():
                script = Path(str(tok))
                break
        if script is None:
            script = Path(__file__).resolve().parents[3] / "scripts" / "run_af3score.py"

        cmd = [
            sys.executable,
            str(script),
            "--input_pdb",
            str(tmp_pdb),
            "--output_dir",
            str(scratch_root),
            "--af3score_repo",
            str(af3_repo),
            "--model_dir",
            str(model_dir),
        ]
        db_dir = cfg.get("db_dir")
        if db_dir:
            db_path = self._resolve_path(ctx, db_dir)
            if db_path:
                cmd.extend(["--db_dir", str(db_path)])
        num_jobs = int(cfg.get("num_jobs") or 1)
        cmd.extend(["--num_jobs", str(num_jobs)])
        if cfg.get("num_workers") is not None:
            cmd.extend(["--num_workers", str(cfg.get("num_workers"))])
        if cfg.get("bucket_default") is not None:
            cmd.extend(["--bucket_default", str(cfg.get("bucket_default"))])
        if cfg.get("num_samples") is not None:
            cmd.extend(["--num_samples", str(cfg.get("num_samples"))])
        if cfg.get("model_seeds") is not None:
            cmd.extend(["--model_seeds", str(cfg.get("model_seeds"))])
        if cfg.get("no_templates"):
            cmd.append("--no_templates")
        if cfg.get("write_cif_model") is not None:
            cmd.extend(["--write_cif_model", str(cfg.get("write_cif_model"))])
        if cfg.get("write_best_model_root"):
            cmd.append("--write_best_model_root")
        if cfg.get("write_ranking_scores_csv"):
            cmd.append("--write_ranking_scores_csv")
        if cfg.get("target_offsets_json"):
            offsets_path = self._resolve_path(ctx, cfg.get("target_offsets_json"))
            if offsets_path:
                cmd.extend(["--target_offsets_json", str(offsets_path)])
        if cfg.get("target_chain"):
            cmd.extend(["--target_chain", str(cfg.get("target_chain"))])

        run_command(
            cmd,
            env=os.environ.copy(),
            cwd=str(scratch_root),
            log_file=self.cfg.get("_log_file"),
            verbose=bool(self.cfg.get("_verbose")),
        )

        def _best_job_cif(job_dir: Path, job_name: str) -> Path | None:
            cand = job_dir / f"{job_name}_model.cif"
            if cand.exists():
                return cand
            seed_cifs = sorted(job_dir.glob("seed-*/*/model.cif"))
            if not seed_cifs:
                seed_cifs = sorted(job_dir.glob("seed-*/model.cif"))
            if seed_cifs:
                return seed_cifs[0]
            return None

        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        out_dir = self.output_dir(ctx)
        metrics_items = out_dir / "metrics_items"
        metrics_items.mkdir(parents=True, exist_ok=True)
        metrics_src = scratch_root / "metrics.csv"
        if not metrics_src.exists():
            raise StepError(f"Missing metrics.csv for {item.id} at {metrics_src}")
        if not is_valid_metrics_shard(metrics_src, expected_desc=str(run_stem)):
            raise StepError(f"Invalid metrics.csv for {item.id} at {metrics_src}")
        metrics_dst = metrics_items / f"{item.id}.csv"
        if metrics_dst.exists() and not is_valid_metrics_shard(metrics_dst, expected_desc=str(item.id)):
            try:
                metrics_dst.unlink()
            except Exception as exc:
                raise StepError(f"Failed to remove invalid promoted metrics shard at {metrics_dst}: {exc}") from exc
        promote_file_atomic(metrics_src, metrics_dst, allow_reuse=allow_reuse)

        job_dir = scratch_root / "af3score_outputs" / run_stem
        cif_src = _best_job_cif(job_dir, run_stem)
        if cif_src is None or not cif_src.exists():
            raise StepError(f"Missing CIF output for {item.id} under {job_dir}")
        cif_dir = out_dir / "cif"
        cif_dir.mkdir(parents=True, exist_ok=True)
        promote_file_atomic(cif_src, cif_dir / f"{item.id}.cif", allow_reuse=allow_reuse)

        # Optional retention: copy heavy/raw runner outputs under output/_optional/... (never required for resume).
        try:
            keep_all = output_mode(ctx) == "full"
        except Exception:
            keep_all = False
        if keep_all or any(should_keep(ctx, k) for k in ["af3score_outputs", "af3_input_batch", "single_chain_cif", "json", "pdbs", "af3score_subprocess_logs"]):
            try:
                opt_root = optional_dir(ctx) / out_dir.name
                raw_root = opt_root / "raw"
                logs_root = opt_root / "logs"
                raw_keys = ["af3score_outputs", "af3_input_batch", "single_chain_cif", "json", "pdbs", "input_pdbs"]
                log_keys = ["af3score_subprocess_logs"]
                for key in raw_keys:
                    if not (keep_all or should_keep(ctx, key)):
                        continue
                    try:
                        promote_tree(scratch_root / key, raw_root / key, allow_reuse=True)
                    except Exception:
                        pass
                for key in log_keys:
                    if not (keep_all or should_keep(ctx, key)):
                        continue
                    try:
                        promote_tree(scratch_root / key, logs_root / key, allow_reuse=True)
                    except Exception:
                        pass
            except Exception:
                pass

        shutil.rmtree(scratch_root, ignore_errors=True)

    def run_batch(self, ctx: StepContext, items: list[WorkItem]) -> dict[str, tuple[str, str | None]]:
        cfg = self._af3score_config()
        af3_repo = self._resolve_path(ctx, cfg.get("af3score_repo")) if cfg else None
        model_dir = self._resolve_path(ctx, cfg.get("model_dir")) if cfg else None
        if not af3_repo or not af3_repo.exists():
            raise StepError("af3score_repo missing or invalid")
        if not model_dir or not model_dir.exists():
            raise StepError("model_dir missing or invalid")

        attempts: list[int] = []
        for it in items:
            try:
                attempts.append(int((it.payload or {}).get("_attempt") or 1))
            except Exception:
                attempts.append(1)
        batch_attempt = max(attempts) if attempts else 1
        batch_id = items[0].id if items else "batch"
        step_name = str(self.cfg.get("name") or self.name)
        batch_dir = step_scratch_dir(ctx, step_name) / f"attempt_{batch_attempt}" / "batches" / f"batch_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        input_dir = batch_dir / "input_pdbs"
        input_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, tuple[str, str | None]] = {}
        valid_items: list[WorkItem] = []
        for item in items:
            pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
            if not pdb_path.exists():
                if self.item_done(ctx, item):
                    results[item.id] = ("done", None)
                else:
                    results[item.id] = ("blocked", f"PDB not found: {pdb_path}")
                continue
            run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
            dst = input_dir / f"{run_stem}.pdb"
            if not dst.exists():
                try:
                    os.link(pdb_path, dst)
                except OSError:
                    shutil.copy2(pdb_path, dst)
            valid_items.append(item)
        if not valid_items:
            shutil.rmtree(batch_dir, ignore_errors=True)
            return results

        script = None
        cmd_cfg = self.cfg.get("command") or []
        if isinstance(cmd_cfg, str):
            cmd_cfg = [cmd_cfg]
        for tok in cmd_cfg:
            if str(tok).endswith("run_af3score.py") and Path(str(tok)).exists():
                script = Path(str(tok))
                break
        if script is None:
            script = Path(__file__).resolve().parents[3] / "scripts" / "run_af3score.py"

        cmd = [
            sys.executable,
            str(script),
            "--input_pdb_dir",
            str(input_dir),
            "--output_dir",
            str(batch_dir),
            "--af3score_repo",
            str(af3_repo),
            "--model_dir",
            str(model_dir),
        ]
        db_dir = cfg.get("db_dir")
        if db_dir:
            db_path = self._resolve_path(ctx, db_dir)
            if db_path:
                cmd.extend(["--db_dir", str(db_path)])
        num_jobs = int(cfg.get("num_jobs") or 1)
        cmd.extend(["--num_jobs", str(num_jobs)])
        if cfg.get("num_workers") is not None:
            cmd.extend(["--num_workers", str(cfg.get("num_workers"))])
        if cfg.get("bucket_default") is not None:
            cmd.extend(["--bucket_default", str(cfg.get("bucket_default"))])
        if cfg.get("num_samples") is not None:
            cmd.extend(["--num_samples", str(cfg.get("num_samples"))])
        if cfg.get("model_seeds") is not None:
            cmd.extend(["--model_seeds", str(cfg.get("model_seeds"))])
        if cfg.get("no_templates"):
            cmd.append("--no_templates")
        if cfg.get("write_cif_model") is not None:
            cmd.extend(["--write_cif_model", str(cfg.get("write_cif_model"))])
        if cfg.get("write_best_model_root"):
            cmd.append("--write_best_model_root")
        if cfg.get("write_ranking_scores_csv"):
            cmd.append("--write_ranking_scores_csv")
        if cfg.get("target_offsets_json"):
            offsets_path = self._resolve_path(ctx, cfg.get("target_offsets_json"))
            if offsets_path:
                cmd.extend(["--target_offsets_json", str(offsets_path)])
        if cfg.get("target_chain"):
            cmd.extend(["--target_chain", str(cfg.get("target_chain"))])

        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        out_dir = self.output_dir(ctx)
        metrics_items = out_dir / "metrics_items"
        metrics_items.mkdir(parents=True, exist_ok=True)
        cif_dir = out_dir / "cif"
        cif_dir.mkdir(parents=True, exist_ok=True)

        def _norm_name(value: str) -> str:
            norm = str(value).strip().lower()
            if norm.endswith(".pdb"):
                norm = norm[:-4]
            return norm

        item_by_stem: dict[str, WorkItem] = {}
        item_by_norm: dict[str, WorkItem] = {}
        attempt_by_id: dict[str, int] = {}
        for item in valid_items:
            run_stem = str((item.payload or {}).get("run_stem") or item.id)
            item_by_stem[run_stem] = item
            item_by_norm[_norm_name(run_stem)] = item
            try:
                attempt_by_id[item.id] = int((item.payload or {}).get("_attempt") or 1)
            except Exception:
                attempt_by_id[item.id] = 1

        def _resolve_af3_python() -> list[str]:
            override = os.environ.get("AF3SCORE_PYTHON")
            if override:
                return shlex.split(override)
            env_name = os.environ.get("AF3SCORE_ENV", "ppiflow-af3score")
            conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
            if conda_exe and env_name:
                return [conda_exe, "run", "-n", env_name, "python"]
            return [sys.executable]

        python_cmd = _resolve_af3_python()
        get_metrics = af3_repo / "04_get_metrics.py"

        def _job_source(job_dir: Path, job_name: str, *, prefer_cif: Path | None = None) -> _AF3JobSource | None:
            return _pick_af3_job_source(job_dir, job_name, prefer_cif=prefer_cif)

        def _input_pdb_for_job(job_name: str, item: WorkItem) -> Path | None:
            cand = input_dir / f"{job_name}.pdb"
            if cand.exists():
                return cand
            raw_path = (item.payload or {}).get("pdb_path")
            if raw_path:
                path = Path(str(raw_path))
                if path.exists():
                    return path
            return None

        def _promote_cif_for_source(source: _AF3JobSource, item: WorkItem) -> bool:
            dst = cif_dir / f"{item.id}.cif"
            if dst.exists():
                try:
                    return files_identical(source.cif_path, dst)
                except Exception:
                    return False
            cif_src = source.cif_path
            if not cif_src.exists():
                return False
            try:
                promote_file_atomic(cif_src, dst, allow_reuse=allow_reuse)
            except Exception:
                return False
            return True

        def _write_metrics_for_source(job_name: str, source: _AF3JobSource, input_pdb: Path, item: WorkItem) -> bool:
            if not get_metrics.exists():
                return False
            tmp_root = batch_dir / ".metrics_tmp" / job_name
            shutil.rmtree(tmp_root, ignore_errors=True)
            input_tmp = tmp_root / "input_pdbs"
            af3_tmp = tmp_root / "af3score_outputs"
            input_tmp.mkdir(parents=True, exist_ok=True)
            af3_tmp.mkdir(parents=True, exist_ok=True)
            seed10_dir = af3_tmp / job_name / "seed-10_sample-0"
            seed10_dir.mkdir(parents=True, exist_ok=True)

            def _link_or_copy(src: Path, dst: Path) -> bool:
                if dst.exists():
                    return True
                try:
                    os.symlink(src, dst)
                    return True
                except Exception:
                    pass
                try:
                    os.link(src, dst)
                    return True
                except Exception:
                    pass
                try:
                    shutil.copy2(src, dst)
                    return True
                except Exception:
                    return False

            if not _link_or_copy(source.summary_conf, seed10_dir / "summary_confidences.json"):
                return False
            if not _link_or_copy(source.full_conf, seed10_dir / "confidences.json"):
                return False
            # Optional but cheap: some metric extractors may read model.cif from the seed directory.
            _link_or_copy(source.cif_path, seed10_dir / "model.cif")
            pdb_dst = input_tmp / f"{job_name}.pdb"
            if not pdb_dst.exists():
                try:
                    os.link(input_pdb, pdb_dst)
                except Exception:
                    shutil.copy2(input_pdb, pdb_dst)
            metrics_csv = tmp_root / "metrics.csv"
            try:
                run_command(
                    python_cmd
                    + [
                        str(get_metrics),
                        "--input_pdb_dir",
                        str(input_tmp),
                        "--af3score_output_dir",
                        str(af3_tmp),
                        "--save_metric_csv",
                        str(metrics_csv),
                    ],
                    env=os.environ.copy(),
                    cwd=str(tmp_root),
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )
            except Exception:
                return False
            if not is_valid_metrics_shard(metrics_csv, expected_desc=str(job_name)):
                return False
            try:
                dst = metrics_items / f"{item.id}.csv"
                if dst.exists() and not is_valid_metrics_shard(dst, expected_desc=str(item.id)):
                    try:
                        dst.unlink()
                    except Exception:
                        return False
                promote_file_atomic(metrics_csv, dst, allow_reuse=allow_reuse)
            except Exception:
                return False
            shutil.rmtree(tmp_root, ignore_errors=True)
            return True

        committed: set[str] = set()
        commit_lock = threading.Lock()

        from ..work_queue import WorkQueue

        wq = WorkQueue(ctx.out_dir, self.name, ctx.work_queue or {})
        forced_failed: dict[str, str] = {}

        def _mark_done(item: WorkItem) -> None:
            try:
                attempt = attempt_by_id.get(item.id, 1)
                wq.mark_done(item.id, attempt)
            except Exception:
                pass
            if ctx.heartbeat:
                try:
                    prog = wq.progress()
                    ctx.heartbeat.update(
                        produced_total=int(prog.get("produced_total", 0)),
                        expected_total=int(prog.get("expected_total", 0)),
                        state=str(prog.get("status") or "running"),
                    )
                except Exception:
                    pass

        def _commit_job(job_name: str, job_dir: Path) -> bool:
            item = item_by_stem.get(job_name) or item_by_norm.get(_norm_name(job_name))
            if not item:
                return False
            if self.item_done(ctx, item):
                committed.add(job_name)
                _mark_done(item)
                return True
            input_pdb = _input_pdb_for_job(job_name, item)
            if input_pdb is None:
                return False
            dst_cif = cif_dir / f"{item.id}.cif"
            prefer = dst_cif if dst_cif.exists() else None
            source = _job_source(job_dir, job_name, prefer_cif=prefer)
            if source is None:
                # If a CIF is already promoted but no seed source matches it byte-identically, we
                # cannot safely recompute/repair metrics without risking model/metrics divergence.
                if prefer is not None:
                    with commit_lock:
                        if job_name in committed:
                            return True
                        attempt = attempt_by_id.get(item.id, 1)
                        msg = (
                            f"AF3Score cannot select a seed source matching existing promoted CIF for {item.id}: "
                            f"{prefer}. (seed-only canonical source selection)"
                        )
                        forced_failed[item.id] = msg
                        wq.mark_failed(item.id, attempt, msg)
                        committed.add(job_name)
                        return True
                return False
            with commit_lock:
                if job_name in committed:
                    return True
                if dst_cif.exists():
                    try:
                        if not files_identical(source.cif_path, dst_cif):
                            attempt = attempt_by_id.get(item.id, 1)
                            msg = (
                                f"AF3Score CIF mismatch for {item.id}: existing {dst_cif} differs from "
                                f"selected seed source {source.cif_path}"
                            )
                            forced_failed[item.id] = msg
                            wq.mark_failed(item.id, attempt, msg)
                            committed.add(job_name)
                            return True
                    except Exception as exc:
                        attempt = attempt_by_id.get(item.id, 1)
                        msg = f"AF3Score CIF compare failed for {item.id}: {exc}"
                        forced_failed[item.id] = msg
                        wq.mark_failed(item.id, attempt, msg)
                        committed.add(job_name)
                        return True
                if not _promote_cif_for_source(source, item):
                    return False
                # Avoid rerunning metrics if a valid shard is already promoted.
                metrics_dst = metrics_items / f"{item.id}.csv"
                if not is_valid_metrics_shard(metrics_dst, expected_desc=str(item.id)):
                    if not _write_metrics_for_source(job_name, source, input_pdb, item):
                        return False
                if self.item_done(ctx, item):
                    committed.add(job_name)
                    _mark_done(item)
                    return True
            return False

        def _scan_batch_outputs(max_per_pass: int | None = None) -> None:
            af3_dir = batch_dir / "af3score_outputs"
            if not af3_dir.exists():
                return
            processed = 0
            for job_dir in sorted(af3_dir.iterdir()):
                if max_per_pass is not None and processed >= max_per_pass:
                    break
                if not job_dir.is_dir():
                    continue
                job_name = job_dir.name
                if job_name not in item_by_stem and _norm_name(job_name) not in item_by_norm:
                    continue
                if job_name in committed:
                    continue
                if _commit_job(job_name, job_dir):
                    processed += 1

        commit_poll = float((ctx.work_queue or {}).get("af3score_commit_poll_s", 5.0) or 5.0)
        commit_batch = int((ctx.work_queue or {}).get("af3score_commit_batch", 25) or 25)
        stop_evt = threading.Event()

        def _commit_loop() -> None:
            while not stop_evt.wait(commit_poll):
                try:
                    _scan_batch_outputs(max_per_pass=commit_batch)
                except Exception:
                    pass

        commit_thread = threading.Thread(target=_commit_loop, daemon=True)
        commit_thread.start()

        err: str | None = None
        if valid_items:
            try:
                run_command(
                    cmd,
                    env=os.environ.copy(),
                    cwd=str(batch_dir),
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )
            except Exception as exc:
                err = str(exc)

        stop_evt.set()
        commit_thread.join(timeout=2.0)
        _scan_batch_outputs()

        # Fallback: split wrapper metrics.csv into per-item shards if any are missing.
        metrics_src = batch_dir / "metrics.csv"
        if metrics_src.exists():
            try:
                import pandas as pd

                df = pd.read_csv(metrics_src)
                for item in valid_items:
                    run_stem = str((item.payload or {}).get("run_stem") or item.id)
                    dst_metrics = metrics_items / f"{item.id}.csv"
                    if dst_metrics.exists():
                        # Repair known-bad promoted shards so retries can recover.
                        if is_valid_metrics_shard(dst_metrics, expected_desc=run_stem):
                            continue
                        try:
                            dst_metrics.unlink()
                        except Exception:
                            continue
                    run_stem_norm = run_stem.lower()
                    if run_stem_norm.endswith(".pdb"):
                        run_stem_norm = run_stem_norm[:-4]
                    mask = None
                    for col in ("description", "name", "model", "pdb_name"):
                        if col in df.columns:
                            series = df[col].astype(str)
                            series_norm = series.str.lower()
                            series_norm = series_norm.str.removesuffix(".pdb")
                            mask = series_norm == run_stem_norm
                            if mask.any():
                                break
                    if mask is not None and mask.any():
                        tmp = batch_dir / f".metrics_item.{item.id}.csv"
                        try:
                            df.loc[mask].to_csv(tmp, index=False)
                            promote_file_atomic(tmp, dst_metrics, allow_reuse=allow_reuse)
                        finally:
                            try:
                                tmp.unlink()
                            except Exception:
                                pass
                    elif len(valid_items) == 1:
                        tmp = batch_dir / f".metrics_item.{item.id}.csv"
                        try:
                            df.to_csv(tmp, index=False)
                            promote_file_atomic(tmp, dst_metrics, allow_reuse=allow_reuse)
                        finally:
                            try:
                                tmp.unlink()
                            except Exception:
                                pass
            except Exception:
                pass

        # Finalize CIF promotion for any remaining items.
        af3_dir = batch_dir / "af3score_outputs"
        if af3_dir.exists():
            for item in valid_items:
                if (cif_dir / f"{item.id}.cif").exists():
                    continue
                run_stem = str((item.payload or {}).get("run_stem") or item.id)
                job_dir = af3_dir / run_stem
                if job_dir.exists():
                    source = _job_source(job_dir, run_stem)
                    if source is not None:
                        _promote_cif_for_source(source, item)

        all_done = all(self.item_done(ctx, item) for item in valid_items)
        if all_done:
            # Optional retention: copy heavy/raw runner outputs under output/_optional/... (never required for resume).
            try:
                keep_all = output_mode(ctx) == "full"
            except Exception:
                keep_all = False
            if keep_all or any(should_keep(ctx, k) for k in ["af3score_outputs", "af3_input_batch", "single_chain_cif", "json", "pdbs", "af3score_subprocess_logs"]):
                try:
                    opt_root = optional_dir(ctx) / out_dir.name
                    raw_root = opt_root / "raw"
                    logs_root = opt_root / "logs"
                    raw_keys = ["af3score_outputs", "af3_input_batch", "single_chain_cif", "json", "pdbs", "input_pdbs"]
                    log_keys = ["af3score_subprocess_logs"]
                    for key in raw_keys:
                        if not (keep_all or should_keep(ctx, key)):
                            continue
                        try:
                            promote_tree(batch_dir / key, raw_root / key, allow_reuse=True)
                        except Exception:
                            pass
                    for key in log_keys:
                        if not (keep_all or should_keep(ctx, key)):
                            continue
                        try:
                            promote_tree(batch_dir / key, logs_root / key, allow_reuse=True)
                        except Exception:
                            pass
                except Exception:
                    pass
            shutil.rmtree(batch_dir, ignore_errors=True)

        for item in valid_items:
            if item.id in forced_failed:
                results[item.id] = ("failed", forced_failed.get(item.id))
            elif self.item_done(ctx, item):
                results[item.id] = ("done", None)
            else:
                results[item.id] = ("failed", err or "missing output")
        return results

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)

        df = None
        metrics_items = out_dir / "metrics_items"
        if metrics_items.exists():
            metrics_paths = sorted(p for p in metrics_items.glob("*.csv"))
            if metrics_paths:
                try:
                    df = pd.concat([pd.read_csv(p) for p in metrics_paths], ignore_index=True)
                except Exception:
                    df = None
        if df is None:
            metrics_path = find_metrics_file(out_dir)
            if not metrics_path:
                return
            df = pd.read_csv(metrics_path)

        # Consolidated metrics.csv (atomic write)
        metrics_csv = out_dir / "metrics.csv"
        tmp_metrics = metrics_csv.parent / f"{metrics_csv.name}.tmp"
        df.to_csv(tmp_metrics, index=False)
        os.replace(tmp_metrics, metrics_csv)

        # Build mapping to upstream PDBs directly from configured input_dir (no metrics_pdbs staging).
        cfg = self._af3score_config()
        input_dir = self.cfg.get("input_dir") or (cfg.get("input_pdb_dir") if cfg else None)
        input_root: Path | None = None
        pdbs: list[Path] = []
        if input_dir:
            input_root = Path(str(input_dir))
            if not input_root.is_absolute():
                input_root = ctx.out_dir / input_root
            if input_root.exists():
                input_root = self._resolve_input_dir(input_root)
                pdbs = collect_pdbs(input_root)

        run_stem_to_pdb: dict[str, Path] = {}
        stem_to_pdb: dict[str, Path] = {}
        if input_root and pdbs:
            try:
                run_stems = compute_run_stems(pdbs, input_root)
                for p in pdbs:
                    run_stem_to_pdb[str(run_stems[p])] = p
                    stem_to_pdb[p.stem.lower()] = p
            except Exception:
                run_stem_to_pdb = {}
                stem_to_pdb = {p.stem.lower(): p for p in pdbs}
        run_stem_to_pdb_lower = {k.lower(): v for k, v in run_stem_to_pdb.items()}

        def _get(row, *keys):
            for k in keys:
                if k in row and pd.notna(row[k]):
                    return row[k]
            return None

        def _get_ci(row, key: str):
            lower_map = {str(c).lower(): c for c in row.index}
            col = lower_map.get(key.lower())
            if col is not None and pd.notna(row[col]):
                return row[col]
            return None

        rows: list[dict[str, Any]] = []
        iptm_global_col: list[float | None] = []
        iptm_binder_target_col: list[float | None] = []
        protocol = ctx.input_data.get("protocol")
        is_antibody = protocol == "antibody"
        target_chain = "B"
        if is_antibody:
            target = ctx.input_data.get("target") or {}
            chains = target.get("chains")
            if isinstance(chains, list) and chains:
                target_chain = str(chains[0])
            elif isinstance(chains, str) and chains:
                target_chain = str(chains)

        step_label = str(self.cfg.get("name") or self.name)
        missing_pdbs: list[str] = []
        for _, row in df.iterrows():
            desc = _get(row, "description", "name", "model", "pdb_name")
            desc = str(desc) if desc is not None else ""
            key = desc.strip().lower()
            if key.endswith(".pdb"):
                key = key[:-4]
            pdb_path = run_stem_to_pdb_lower.get(key) or stem_to_pdb.get(key)
            if desc and pdb_path is None:
                missing_pdbs.append(desc)

            design_id = extract_design_id(desc)
            iptm_global = _get(row, "iptm", "ipTM", "AF3Score_interchain_iptm", "AF3Score_chain_iptm")
            iptm_binder_target = None
            if is_antibody:
                chain_target_iptm = _get_ci(row, f"chain_{target_chain}_iptm")
                if chain_target_iptm is not None:
                    iptm_binder_target = float(chain_target_iptm)
                else:
                    raise StepError(
                        "AF3Score metrics missing antibody binder-target iptm. "
                        "Require chain_<target>_iptm (e.g., chain_B_iptm)."
                    )
                iptm = iptm_binder_target
            else:
                iptm = iptm_global
            ptm = _get(row, "ptm", "pTM", "ptm_A", "ptm_B")

            iptm_global_col.append(float(iptm_global) if iptm_global is not None else None)
            iptm_binder_target_col.append(
                float(iptm_binder_target) if iptm_binder_target is not None else None
            )
            rows.append({
                # Internal-only (not written to the manifest CSV): used for filtered_pdbs naming.
                "run_stem": desc,
                "design_id": design_id,
                "structure_id": structure_id_from_name(desc),
                "iptm": float(iptm) if iptm is not None else None,
                "iptm_binder_target": (
                    float(iptm_binder_target) if iptm_binder_target is not None else None
                ),
                "iptm_global": float(iptm_global) if iptm_global is not None else None,
                "ptm": float(ptm) if ptm is not None else None,
                "pdb_path": str(pdb_path) if pdb_path else None,
            })

        if missing_pdbs:
            sample = ", ".join(missing_pdbs[:5])
            print(
                f"[{step_label}] WARN missing {len(missing_pdbs)} upstream PDBs for AF3Score rows "
                f"(examples: {sample}).",
                flush=True,
            )

        # Write derived metrics with binder-target ipTM if needed (atomic write).
        df_ppi = df.copy()
        df_ppi["iptm_global"] = iptm_global_col
        df_ppi["iptm_binder_target"] = iptm_binder_target_col
        metrics_ppiflow = out_dir / "metrics_ppiflow.csv"
        tmp_ppi = metrics_ppiflow.parent / f"{metrics_ppiflow.name}.tmp"
        df_ppi.to_csv(tmp_ppi, index=False)
        os.replace(tmp_ppi, metrics_ppiflow)

        if not rows:
            return

        filters = (ctx.input_data.get("filters") or {}).get("af3score") or {}
        if self.name == "af3score2" or self.cfg.get("name") == "af3score2":
            iptm_min = float((filters.get("round2") or {}).get("iptm_min") or 0)
            ptm_min = float((filters.get("round2") or {}).get("ptm_min") or 0)
        else:
            iptm_min = float((filters.get("round1") or {}).get("iptm_min") or 0)
            ptm_val = (filters.get("round1") or {}).get("ptm_min")
            ptm_min = float(ptm_val or 0) if ptm_val is not None else 0.0

        for r in rows:
            iptm_ok = r.get("iptm") is not None and float(r.get("iptm") or 0) >= iptm_min
            ptm_ok = True
            if ptm_min:
                ptm_ok = r.get("ptm") is not None and float(r.get("ptm") or 0) >= ptm_min
            r["passed_filter"] = bool(iptm_ok and ptm_ok)

        top_k = None
        if self.name == "af3score2" or self.cfg.get("name") == "af3score2":
            top_k = (filters.get("round2") or {}).get("top_k")
        else:
            top_k = (filters.get("round1") or {}).get("top_k")
        try:
            top_k = int(top_k) if top_k is not None else None
        except Exception:
            top_k = None

        filtered_rows = [r for r in rows if r.get("passed_filter")]
        if top_k is not None and top_k > 0:
            filtered_rows = sorted(
                filtered_rows,
                key=lambda r: (r.get("iptm") is not None, float(r.get("iptm") or 0)),
                reverse=True,
            )
            keep_ids = set()
            for r in filtered_rows[:top_k]:
                keep_ids.add((r.get("design_id"), r.get("structure_id")))
            for r in rows:
                r["passed_top_k"] = (r.get("design_id"), r.get("structure_id")) in keep_ids
        else:
            for r in rows:
                r["passed_top_k"] = r.get("passed_filter")

        # Rebuild filtered_pdbs/ deterministically to avoid downstream picking up stale files.
        filtered_dir = out_dir / "filtered_pdbs"
        if filtered_dir.exists():
            shutil.rmtree(filtered_dir, ignore_errors=True)
        filtered_dir.mkdir(parents=True, exist_ok=True)
        expected_passing = 0
        missing_passing: list[str] = []
        passing = 0
        for r in rows:
            if not r.get("passed_filter"):
                continue
            expected_passing += 1
            pdb_path = r.get("pdb_path")
            run_stem = str(r.get("run_stem") or "").strip() or "<missing>"
            if not pdb_path:
                missing_passing.append(run_stem)
                continue
            src = Path(str(pdb_path))
            if not src.exists():
                missing_passing.append(run_stem)
                continue
            if run_stem == "<missing>":
                run_stem = src.stem
            if run_stem.lower().endswith(".pdb"):
                run_stem = run_stem[:-4]
            run_stem = run_stem.replace("/", "_").replace("\\", "_")
            dst = filtered_dir / f"{run_stem}.pdb"
            try:
                promote_file_atomic(src, dst, allow_reuse=True)
            except Exception as exc:
                raise StepError(f"Failed to materialize filtered_pdbs for {run_stem}: {exc}") from exc
            passing += 1

        if expected_passing == 0:
            try:
                payload = {
                    "step": step_label,
                    "reason": "no_passing_structures",
                    "total_rows": int(len(rows)),
                    "iptm_min": float(iptm_min),
                    "ptm_min": float(ptm_min),
                    "top_k": int(top_k) if top_k is not None else None,
                }
                (out_dir / "no_candidates.json").write_text(json.dumps(payload, indent=2) + "\n")
            except Exception:
                pass
            raise StepError(
                "AF3Score produced 0 passing structures after filtering; adjust thresholds or disable filtering."
            )
        if passing != expected_passing:
            sample = ", ".join(missing_passing[:5])
            msg = (
                f"AF3Score expected {expected_passing} passing structures but only materialized "
                f"{passing} into filtered_pdbs; missing PDBs for {len(missing_passing)} passing rows "
                f"(examples: {sample})."
            )
            raise StepError(msg)

        # Write manifest atomically.
        manifest_path = self.manifest_path(ctx)
        tmp_manifest = manifest_path.parent / f"{manifest_path.name}.tmp"
        write_csv(
            tmp_manifest,
            rows,
            [
                "design_id",
                "structure_id",
                "iptm",
                "iptm_binder_target",
                "iptm_global",
                "ptm",
                "pdb_path",
                "passed_filter",
                "passed_top_k",
            ],
        )
        os.replace(tmp_manifest, manifest_path)

    def outputs_complete(self, ctx: StepContext) -> bool:
        # Item outputs + metadata must be valid, and downstream-critical derived outputs must exist.
        if not super().outputs_complete(ctx):
            return False
        out_dir = self._resolve_output_dir_path(ctx)
        if not (out_dir / "metrics.csv").exists():
            return False
        if not (out_dir / "metrics_ppiflow.csv").exists():
            return False
        filtered_dir = out_dir / "filtered_pdbs"
        if not filtered_dir.exists() or not list(filtered_dir.glob("*.pdb")):
            return False
        return True

    def _finalize_work_queue_outputs(self, ctx: StepContext, wq, *, items: list[Any], allow_failures: bool) -> None:
        # AF3 finalize must fail fast if derived outputs cannot be materialized; do not write metadata
        # unless write_manifest() succeeds.
        try:
            counts = wq.counts()
        except Exception:
            counts = None
        if counts and (counts.get("pending", 0) != 0 or counts.get("running", 0) != 0):
            return
        if not wq.acquire_leader():
            return
        try:
            if self.cfg.get("manifest"):
                # Strict: propagate exceptions (e.g. zero passing structures).
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

    def scan_done(self, ctx: StepContext) -> set[int]:
        done = super().scan_done(ctx)
        if done:
            return done
        out_dir = self.output_dir(ctx)
        metrics = out_dir / "metrics.csv"
        if metrics.exists():
            return set(range(self.expected_total(ctx)))
        return set()


class RosettaInterfaceStep(ExternalCommandStep):
    name = "rosetta_interface"
    stage = "rosetta"

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        residue_csv = out_dir / "residue_energy.csv"
        if not residue_csv.exists():
            return
        try:
            df = pd.read_csv(residue_csv)
        except Exception:
            return
        rows = []
        for _, row in df.iterrows():
            name = str(row.get("pdbname") or Path(str(row.get("pdbpath", ""))).stem)
            rows.append({
                "design_id": extract_design_id(name),
                "structure_id": structure_id_from_name(name),
                "residue_energy_csv": str(residue_csv),
            })
        if not rows:
            return
        write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "residue_energy_csv"])

    def scan_done(self, ctx: StepContext) -> set[int]:
        done = super().scan_done(ctx)
        if done:
            return done
        out_dir = self.output_dir(ctx)
        if (out_dir / "residue_energy.csv").exists():
            return set(range(self.expected_total(ctx)))
        return set()


class RelaxStep(ExternalCommandStep):
    name = "relax"
    stage = "rosetta"

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        rows = []
        for fp in sorted(out_dir.rglob("*.pdb")):
            if is_ignored_path(fp):
                continue
            rows.append({
                "design_id": extract_design_id(fp.stem),
                "structure_id": structure_id_from_name(fp.stem),
                "pdb_path": str(fp),
            })
        if not rows:
            return
        write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "pdb_path"])

    def scan_done(self, ctx: StepContext) -> set[int]:
        done = super().scan_done(ctx)
        if done:
            return done
        out_dir = self.output_dir(ctx)
        for fp in out_dir.rglob("*.pdb"):
            if is_ignored_path(fp):
                continue
            return set(range(self.expected_total(ctx)))
        return set()


class AF3RefoldStep(ExternalCommandStep):
    name = "af3_refold"
    stage = "score"

    supports_work_queue = True
    work_queue_mode = "items"
    # Drain per-worker batches to avoid repeated model reloads.
    per_worker_batch = True
    # 0 means "claim all available items for this worker".
    batch_size = 0

    def _af3score_config(self) -> dict:
        cfg = dict(self.cfg.get("af3score") or {})
        if cfg:
            return cfg
        cmd = self.cfg.get("command") or []
        if isinstance(cmd, str):
            cmd = [cmd]
        out: dict[str, object] = {}
        for idx, token in enumerate(cmd):
            if token in {
                "--input_pdb_dir",
                "--output_dir",
                "--af3score_repo",
                "--model_dir",
                "--db_dir",
                "--num_jobs",
                "--num_workers",
                "--bucket_default",
                "--num_samples",
                "--model_seeds",
                "--write_cif_model",
                "--export_pdb_dir",
                "--export_cif_dir",
                "--target_offsets_json",
                "--target_chain",
            }:
                if idx + 1 < len(cmd):
                    out[token.lstrip("-")] = cmd[idx + 1]
            elif token == "--no_templates":
                out["no_templates"] = True
            elif token == "--write_best_model_root":
                out["write_best_model_root"] = True
            elif token == "--write_ranking_scores_csv":
                out["write_ranking_scores_csv"] = True
        return out

    def _resolve_path(self, ctx: StepContext, value: str | None) -> Path | None:
        if not value:
            return None
        p = Path(str(value))
        if p.is_absolute():
            return p
        return ctx.out_dir / p

    def _resolve_input_dir(self, input_dir: Path) -> Path:
        if (input_dir / "run_1").exists():
            return input_dir / "run_1"
        return input_dir

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        return self.list_items(ctx, readonly=False)

    def list_items(self, ctx: StepContext, *, readonly: bool = False) -> list[WorkItem]:
        _ = readonly  # list_items must be read-only for rebuild/output checks.
        cfg = self._af3score_config()
        input_dir = self.cfg.get("input_dir") or (cfg.get("input_pdb_dir") if cfg else None)
        if not input_dir:
            raise StepError("af3_refold input_dir missing or invalid")
        input_pdb_dir = Path(str(input_dir))
        if not input_pdb_dir.is_absolute():
            input_pdb_dir = ctx.out_dir / input_pdb_dir
        if not input_pdb_dir.exists():
            raise StepError(f"af3_refold input_dir missing or invalid: {input_pdb_dir}")
        input_pdb_dir = self._resolve_input_dir(input_pdb_dir)

        pdbs = collect_pdbs(input_pdb_dir)
        if not pdbs:
            raise StepError(f"No PDBs found for af3_refold in {input_pdb_dir}")
        run_stems = compute_run_stems(pdbs, input_pdb_dir)

        items: list[WorkItem] = []
        for pdb_path in pdbs:
            run_stem = run_stems[pdb_path]
            items.append(
                WorkItem(
                    id=run_stem,
                    payload={
                        "pdb_path": str(pdb_path),
                        "pdb_name": pdb_path.stem,
                        "run_stem": run_stem,
                    },
                )
            )
        return items

    def item_done(self, ctx: StepContext, item: WorkItem) -> bool:
        out_dir = self.output_dir(ctx)
        metrics_path = out_dir / "metrics_items" / f"{item.id}.csv"
        cif_path = out_dir / "cif" / f"{item.id}.cif"
        pdb_path = out_dir / "pdbs" / f"{item.id}.pdb"
        return (
            is_valid_metrics_shard(metrics_path, expected_desc=str(item.id))
            and cif_path.exists()
            and pdb_path.exists()
        )

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        cfg = self._af3score_config()
        af3_repo = self._resolve_path(ctx, cfg.get("af3score_repo")) if cfg else None
        model_dir = self._resolve_path(ctx, cfg.get("model_dir")) if cfg else None
        if not af3_repo or not af3_repo.exists():
            raise StepError("af3score_repo missing or invalid")
        if not model_dir or not model_dir.exists():
            raise StepError("model_dir missing or invalid")

        pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
        try:
            attempt = int((item.payload or {}).get("_attempt") or 1)
        except Exception:
            attempt = 1

        scratch_root = step_scratch_dir(ctx, self.name) / f"attempt_{attempt}" / "items" / item.id
        scratch_root.mkdir(parents=True, exist_ok=True)
        tmp_pdb = scratch_root / f"{run_stem}.pdb"
        if not tmp_pdb.exists():
            try:
                os.link(pdb_path, tmp_pdb)
            except OSError:
                shutil.copy2(pdb_path, tmp_pdb)

        script = None
        cmd_cfg = self.cfg.get("command") or []
        if isinstance(cmd_cfg, str):
            cmd_cfg = [cmd_cfg]
        for tok in cmd_cfg:
            if str(tok).endswith("run_af3score.py") and Path(str(tok)).exists():
                script = Path(str(tok))
                break
        if script is None:
            script = Path(__file__).resolve().parents[3] / "scripts" / "run_af3score.py"

        cmd = [
            sys.executable,
            str(script),
            "--input_pdb",
            str(tmp_pdb),
            "--output_dir",
            str(scratch_root),
            "--af3score_repo",
            str(af3_repo),
            "--model_dir",
            str(model_dir),
        ]
        db_dir = cfg.get("db_dir")
        if db_dir:
            db_path = self._resolve_path(ctx, db_dir)
            if db_path:
                cmd.extend(["--db_dir", str(db_path)])
        num_jobs = int(cfg.get("num_jobs") or 1)
        cmd.extend(["--num_jobs", str(num_jobs)])
        if cfg.get("num_workers") is not None:
            cmd.extend(["--num_workers", str(cfg.get("num_workers"))])
        if cfg.get("bucket_default") is not None:
            cmd.extend(["--bucket_default", str(cfg.get("bucket_default"))])
        if cfg.get("num_samples") is not None:
            cmd.extend(["--num_samples", str(cfg.get("num_samples"))])
        if cfg.get("model_seeds") is not None:
            cmd.extend(["--model_seeds", str(cfg.get("model_seeds"))])
        if cfg.get("no_templates"):
            cmd.append("--no_templates")
        if cfg.get("write_cif_model") is not None:
            cmd.extend(["--write_cif_model", str(cfg.get("write_cif_model"))])
        if cfg.get("write_best_model_root"):
            cmd.append("--write_best_model_root")
        if cfg.get("write_ranking_scores_csv"):
            cmd.append("--write_ranking_scores_csv")

        run_command(
            cmd,
            env=os.environ.copy(),
            cwd=str(scratch_root),
            log_file=self.cfg.get("_log_file"),
            verbose=bool(self.cfg.get("_verbose")),
        )

        def _best_job_cif(job_dir: Path, job_name: str) -> Path | None:
            cand = job_dir / f"{job_name}_model.cif"
            if cand.exists():
                return cand
            seed_cifs = sorted(job_dir.glob("seed-*/*/model.cif"))
            if not seed_cifs:
                seed_cifs = sorted(job_dir.glob("seed-*/model.cif"))
            if seed_cifs:
                return seed_cifs[0]
            return None

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

        def _convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
            try:
                from Bio.PDB import MMCIFParser, PDBIO
            except Exception:
                return False
            try:
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("model", str(cif_path))
                io = PDBIO()
                io.set_structure(structure)
                io.save(str(pdb_path))
                return True
            except Exception:
                return False

        def _renumber_chain_with_offsets(pdb_path: Path, chain_id: str, mapping: list[int]) -> bool:
            try:
                from Bio.PDB import PDBParser, PDBIO
            except Exception:
                return False
            try:
                parser = PDBParser(QUIET=True)
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
                io = PDBIO()
                io.set_structure(structure)
                io.save(str(pdb_path))
                return True
            except Exception:
                return False

        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        out_dir = self.output_dir(ctx)
        metrics_items = out_dir / "metrics_items"
        metrics_items.mkdir(parents=True, exist_ok=True)
        cif_dir = out_dir / "cif"
        cif_dir.mkdir(parents=True, exist_ok=True)
        pdbs_dir = out_dir / "pdbs"
        pdbs_dir.mkdir(parents=True, exist_ok=True)

        metrics_src = scratch_root / "metrics.csv"
        if not metrics_src.exists():
            raise StepError(f"Missing metrics.csv for {item.id} at {metrics_src}")
        if not is_valid_metrics_shard(metrics_src, expected_desc=str(run_stem)):
            raise StepError(f"Invalid metrics.csv for {item.id} at {metrics_src}")
        metrics_dst = metrics_items / f"{item.id}.csv"
        if metrics_dst.exists() and not is_valid_metrics_shard(metrics_dst, expected_desc=str(item.id)):
            try:
                metrics_dst.unlink()
            except Exception as exc:
                raise StepError(f"Failed to remove invalid promoted metrics shard at {metrics_dst}: {exc}") from exc
        promote_file_atomic(metrics_src, metrics_dst, allow_reuse=allow_reuse)

        job_dir = scratch_root / "af3score_outputs" / run_stem
        cif_src = _best_job_cif(job_dir, run_stem)
        if cif_src is None or not cif_src.exists():
            raise StepError(f"Missing CIF output for {item.id} under {job_dir}")
        promote_file_atomic(cif_src, cif_dir / f"{item.id}.cif", allow_reuse=allow_reuse)

        # Convert CIF -> PDB in scratch, optionally renumber chain offsets, then promote.
        cfg_target_chain = str(cfg.get("target_chain") or "B")
        offset_map = None
        offsets_val = cfg.get("target_offsets_json")
        if offsets_val:
            offsets_path = self._resolve_path(ctx, str(offsets_val))
            if offsets_path and offsets_path.exists():
                offset_map = _load_chain_offset_map(offsets_path)
        scratch_pdb = scratch_root / f"{item.id}.pdb"
        if not _convert_cif_to_pdb(cif_src, scratch_pdb):
            raise StepError(f"Failed to convert CIF to PDB for {item.id}")
        if offset_map:
            _renumber_chain_with_offsets(scratch_pdb, cfg_target_chain, offset_map)
        promote_file_atomic(scratch_pdb, pdbs_dir / f"{item.id}.pdb", allow_reuse=allow_reuse)

        # Optional retention: copy heavy/raw runner outputs under output/_optional/... (never required for resume).
        try:
            keep_all = output_mode(ctx) == "full"
        except Exception:
            keep_all = False
        if keep_all or any(should_keep(ctx, k) for k in ["af3score_outputs", "af3_input_batch", "single_chain_cif", "json", "af3score_subprocess_logs"]):
            try:
                opt_root = optional_dir(ctx) / out_dir.name
                raw_root = opt_root / "raw"
                logs_root = opt_root / "logs"
                raw_keys = ["af3score_outputs", "af3_input_batch", "single_chain_cif", "json", "input_pdbs"]
                log_keys = ["af3score_subprocess_logs"]
                for key in raw_keys:
                    if not (keep_all or should_keep(ctx, key)):
                        continue
                    try:
                        promote_tree(scratch_root / key, raw_root / key, allow_reuse=True)
                    except Exception:
                        pass
                for key in log_keys:
                    if not (keep_all or should_keep(ctx, key)):
                        continue
                    try:
                        promote_tree(scratch_root / key, logs_root / key, allow_reuse=True)
                    except Exception:
                        pass
            except Exception:
                pass

        shutil.rmtree(scratch_root, ignore_errors=True)

    def run_batch(self, ctx: StepContext, items: list[WorkItem]) -> dict[str, tuple[str, str | None]]:
        cfg = self._af3score_config()
        af3_repo = self._resolve_path(ctx, cfg.get("af3score_repo")) if cfg else None
        model_dir = self._resolve_path(ctx, cfg.get("model_dir")) if cfg else None
        if not af3_repo or not af3_repo.exists():
            raise StepError("af3score_repo missing or invalid")
        if not model_dir or not model_dir.exists():
            raise StepError("model_dir missing or invalid")

        attempts: list[int] = []
        for it in items:
            try:
                attempts.append(int((it.payload or {}).get("_attempt") or 1))
            except Exception:
                attempts.append(1)
        batch_attempt = max(attempts) if attempts else 1
        batch_id = items[0].id if items else "batch"
        batch_dir = step_scratch_dir(ctx, self.name) / f"attempt_{batch_attempt}" / "batches" / f"batch_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        input_dir = batch_dir / "input_pdbs"
        input_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, tuple[str, str | None]] = {}
        valid_items: list[WorkItem] = []
        for item in items:
            pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
            if not pdb_path.exists():
                if self.item_done(ctx, item):
                    results[item.id] = ("done", None)
                else:
                    results[item.id] = ("blocked", f"PDB not found: {pdb_path}")
                continue
            run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
            dst = input_dir / f"{run_stem}.pdb"
            if not dst.exists():
                try:
                    os.link(pdb_path, dst)
                except OSError:
                    shutil.copy2(pdb_path, dst)
            valid_items.append(item)
        if not valid_items:
            shutil.rmtree(batch_dir, ignore_errors=True)
            return results

        script = None
        cmd_cfg = self.cfg.get("command") or []
        if isinstance(cmd_cfg, str):
            cmd_cfg = [cmd_cfg]
        for tok in cmd_cfg:
            if str(tok).endswith("run_af3score.py") and Path(str(tok)).exists():
                script = Path(str(tok))
                break
        if script is None:
            script = Path(__file__).resolve().parents[3] / "scripts" / "run_af3score.py"

        cmd = [
            sys.executable,
            str(script),
            "--input_pdb_dir",
            str(input_dir),
            "--output_dir",
            str(batch_dir),
            "--af3score_repo",
            str(af3_repo),
            "--model_dir",
            str(model_dir),
        ]
        db_dir = cfg.get("db_dir")
        if db_dir:
            db_path = self._resolve_path(ctx, db_dir)
            if db_path:
                cmd.extend(["--db_dir", str(db_path)])
        num_jobs = int(cfg.get("num_jobs") or 1)
        cmd.extend(["--num_jobs", str(num_jobs)])
        if cfg.get("num_workers") is not None:
            cmd.extend(["--num_workers", str(cfg.get("num_workers"))])
        if cfg.get("bucket_default") is not None:
            cmd.extend(["--bucket_default", str(cfg.get("bucket_default"))])
        if cfg.get("num_samples") is not None:
            cmd.extend(["--num_samples", str(cfg.get("num_samples"))])
        if cfg.get("model_seeds") is not None:
            cmd.extend(["--model_seeds", str(cfg.get("model_seeds"))])
        if cfg.get("no_templates"):
            cmd.append("--no_templates")
        if cfg.get("write_cif_model") is not None:
            cmd.extend(["--write_cif_model", str(cfg.get("write_cif_model"))])
        if cfg.get("write_best_model_root"):
            cmd.append("--write_best_model_root")
        if cfg.get("write_ranking_scores_csv"):
            cmd.append("--write_ranking_scores_csv")

        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        out_dir = self.output_dir(ctx)
        metrics_items = out_dir / "metrics_items"
        metrics_items.mkdir(parents=True, exist_ok=True)
        cif_dir = out_dir / "cif"
        cif_dir.mkdir(parents=True, exist_ok=True)
        pdbs_dir = out_dir / "pdbs"
        pdbs_dir.mkdir(parents=True, exist_ok=True)

        def _norm_name(value: str) -> str:
            norm = str(value).strip().lower()
            if norm.endswith(".pdb"):
                norm = norm[:-4]
            return norm

        item_by_stem: dict[str, WorkItem] = {}
        item_by_norm: dict[str, WorkItem] = {}
        attempt_by_id: dict[str, int] = {}
        for item in valid_items:
            run_stem = str((item.payload or {}).get("run_stem") or item.id)
            item_by_stem[run_stem] = item
            item_by_norm[_norm_name(run_stem)] = item
            try:
                attempt_by_id[item.id] = int((item.payload or {}).get("_attempt") or 1)
            except Exception:
                attempt_by_id[item.id] = 1

        def _resolve_af3_python() -> list[str]:
            override = os.environ.get("AF3SCORE_PYTHON")
            if override:
                return shlex.split(override)
            env_name = os.environ.get("AF3SCORE_ENV", "ppiflow-af3score")
            conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
            if conda_exe and env_name:
                return [conda_exe, "run", "-n", env_name, "python"]
            return [sys.executable]

        python_cmd = _resolve_af3_python()
        get_metrics = af3_repo / "04_get_metrics.py"

        def _job_source(job_dir: Path, job_name: str, *, prefer_cif: Path | None = None) -> _AF3JobSource | None:
            return _pick_af3_job_source(job_dir, job_name, prefer_cif=prefer_cif)

        def _input_pdb_for_job(job_name: str, item: WorkItem) -> Path | None:
            cand = input_dir / f"{job_name}.pdb"
            if cand.exists():
                return cand
            raw_path = (item.payload or {}).get("pdb_path")
            if raw_path:
                path = Path(str(raw_path))
                if path.exists():
                    return path
            return None

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

        def _convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
            try:
                from Bio.PDB import MMCIFParser, PDBIO
            except Exception:
                return False
            try:
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("model", str(cif_path))
                io = PDBIO()
                io.set_structure(structure)
                io.save(str(pdb_path))
                return True
            except Exception:
                return False

        def _renumber_chain_with_offsets(pdb_path: Path, chain_id: str, mapping: list[int]) -> bool:
            try:
                from Bio.PDB import PDBParser, PDBIO
            except Exception:
                return False
            try:
                parser = PDBParser(QUIET=True)
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
                io = PDBIO()
                io.set_structure(structure)
                io.save(str(pdb_path))
                return True
            except Exception:
                return False

        cfg_target_chain = str(cfg.get("target_chain") or "B")
        offset_map = None
        offsets_val = cfg.get("target_offsets_json")
        if offsets_val:
            offsets_path = self._resolve_path(ctx, str(offsets_val))
            if offsets_path and offsets_path.exists():
                offset_map = _load_chain_offset_map(offsets_path)

        def _promote_cif_for_source(source: _AF3JobSource, item: WorkItem) -> bool:
            dst = cif_dir / f"{item.id}.cif"
            if dst.exists():
                try:
                    return files_identical(source.cif_path, dst)
                except Exception:
                    return False
            cif_src = source.cif_path
            if not cif_src.exists():
                return False
            try:
                promote_file_atomic(cif_src, dst, allow_reuse=allow_reuse)
            except Exception:
                return False
            return True

        def _promote_pdb_for_source(source: _AF3JobSource, item: WorkItem) -> bool:
            dst = pdbs_dir / f"{item.id}.pdb"
            cif_src = source.cif_path
            if not cif_src.exists():
                return False
            tmp_dir = batch_dir / ".pdb_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_pdb = tmp_dir / f"{item.id}.pdb"
            try:
                if not _convert_cif_to_pdb(cif_src, tmp_pdb):
                    return False
                if offset_map:
                    _renumber_chain_with_offsets(tmp_pdb, cfg_target_chain, offset_map)
                # If a PDB already exists, verify it matches the canonical CIF-derived PDB. If not,
                # treat it as a repairable derived artifact and replace it.
                if dst.exists():
                    try:
                        if files_identical(tmp_pdb, dst):
                            return True
                    except Exception:
                        # Fall through to repair attempt below.
                        pass
                    try:
                        dst.unlink()
                    except Exception:
                        return False
                try:
                    promote_file_atomic(tmp_pdb, dst, allow_reuse=allow_reuse)
                except Exception:
                    # Another worker may have raced to install a matching file.
                    try:
                        return dst.exists() and files_identical(tmp_pdb, dst)
                    except Exception:
                        return False
                return True
            finally:
                try:
                    tmp_pdb.unlink()
                except Exception:
                    pass

        def _write_metrics_for_source(job_name: str, source: _AF3JobSource, input_pdb: Path, item: WorkItem) -> bool:
            if not get_metrics.exists():
                return False
            tmp_root = batch_dir / ".metrics_tmp" / job_name
            shutil.rmtree(tmp_root, ignore_errors=True)
            input_tmp = tmp_root / "input_pdbs"
            af3_tmp = tmp_root / "af3score_outputs"
            input_tmp.mkdir(parents=True, exist_ok=True)
            af3_tmp.mkdir(parents=True, exist_ok=True)
            seed10_dir = af3_tmp / job_name / "seed-10_sample-0"
            seed10_dir.mkdir(parents=True, exist_ok=True)

            def _link_or_copy(src: Path, dst: Path) -> bool:
                if dst.exists():
                    return True
                try:
                    os.symlink(src, dst)
                    return True
                except Exception:
                    pass
                try:
                    os.link(src, dst)
                    return True
                except Exception:
                    pass
                try:
                    shutil.copy2(src, dst)
                    return True
                except Exception:
                    return False

            if not _link_or_copy(source.summary_conf, seed10_dir / "summary_confidences.json"):
                return False
            if not _link_or_copy(source.full_conf, seed10_dir / "confidences.json"):
                return False
            # Optional but cheap: some metric extractors may read model.cif from the seed directory.
            _link_or_copy(source.cif_path, seed10_dir / "model.cif")
            pdb_dst = input_tmp / f"{job_name}.pdb"
            if not pdb_dst.exists():
                try:
                    os.link(input_pdb, pdb_dst)
                except Exception:
                    shutil.copy2(input_pdb, pdb_dst)
            metrics_csv = tmp_root / "metrics.csv"
            try:
                run_command(
                    python_cmd
                    + [
                        str(get_metrics),
                        "--input_pdb_dir",
                        str(input_tmp),
                        "--af3score_output_dir",
                        str(af3_tmp),
                        "--save_metric_csv",
                        str(metrics_csv),
                    ],
                    env=os.environ.copy(),
                    cwd=str(tmp_root),
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )
            except Exception:
                return False
            if not is_valid_metrics_shard(metrics_csv, expected_desc=str(job_name)):
                return False
            try:
                dst = metrics_items / f"{item.id}.csv"
                if dst.exists() and not is_valid_metrics_shard(dst, expected_desc=str(item.id)):
                    try:
                        dst.unlink()
                    except Exception:
                        return False
                promote_file_atomic(metrics_csv, dst, allow_reuse=allow_reuse)
            except Exception:
                return False
            shutil.rmtree(tmp_root, ignore_errors=True)
            return True

        committed: set[str] = set()
        commit_lock = threading.Lock()

        from ..work_queue import WorkQueue

        wq = WorkQueue(ctx.out_dir, self.name, ctx.work_queue or {})
        forced_failed: dict[str, str] = {}

        def _mark_done(item: WorkItem) -> None:
            try:
                attempt = attempt_by_id.get(item.id, 1)
                wq.mark_done(item.id, attempt)
            except Exception:
                pass
            if ctx.heartbeat:
                try:
                    prog = wq.progress()
                    ctx.heartbeat.update(
                        produced_total=int(prog.get("produced_total", 0)),
                        expected_total=int(prog.get("expected_total", 0)),
                        state=str(prog.get("status") or "running"),
                    )
                except Exception:
                    pass

        def _commit_job(job_name: str, job_dir: Path) -> bool:
            item = item_by_stem.get(job_name) or item_by_norm.get(_norm_name(job_name))
            if not item:
                return False
            if self.item_done(ctx, item):
                committed.add(job_name)
                _mark_done(item)
                return True
            input_pdb = _input_pdb_for_job(job_name, item)
            if input_pdb is None:
                return False
            dst_cif = cif_dir / f"{item.id}.cif"
            prefer = dst_cif if dst_cif.exists() else None
            source = _job_source(job_dir, job_name, prefer_cif=prefer)
            if source is None:
                if prefer is not None:
                    with commit_lock:
                        if job_name in committed:
                            return True
                        attempt = attempt_by_id.get(item.id, 1)
                        msg = (
                            f"AF3Refold cannot select a seed source matching existing promoted CIF for {item.id}: "
                            f"{prefer}. (seed-only canonical source selection)"
                        )
                        forced_failed[item.id] = msg
                        wq.mark_failed(item.id, attempt, msg)
                        committed.add(job_name)
                        return True
                return False
            with commit_lock:
                if job_name in committed:
                    return True
                if dst_cif.exists():
                    try:
                        if not files_identical(source.cif_path, dst_cif):
                            attempt = attempt_by_id.get(item.id, 1)
                            msg = (
                                f"AF3Refold CIF mismatch for {item.id}: existing {dst_cif} differs from "
                                f"selected seed source {source.cif_path}"
                            )
                            forced_failed[item.id] = msg
                            wq.mark_failed(item.id, attempt, msg)
                            committed.add(job_name)
                            return True
                    except Exception as exc:
                        attempt = attempt_by_id.get(item.id, 1)
                        msg = f"AF3Refold CIF compare failed for {item.id}: {exc}"
                        forced_failed[item.id] = msg
                        wq.mark_failed(item.id, attempt, msg)
                        committed.add(job_name)
                        return True
                if not _promote_cif_for_source(source, item):
                    return False
                if not _promote_pdb_for_source(source, item):
                    return False
                # Avoid rerunning metrics if a valid shard is already promoted.
                metrics_dst = metrics_items / f"{item.id}.csv"
                if not is_valid_metrics_shard(metrics_dst, expected_desc=str(item.id)):
                    if not _write_metrics_for_source(job_name, source, input_pdb, item):
                        return False
                if self.item_done(ctx, item):
                    committed.add(job_name)
                    _mark_done(item)
                    return True
            return False

        def _scan_batch_outputs(max_per_pass: int | None = None) -> None:
            af3_dir = batch_dir / "af3score_outputs"
            if not af3_dir.exists():
                return
            processed = 0
            for job_dir in sorted(af3_dir.iterdir()):
                if max_per_pass is not None and processed >= max_per_pass:
                    break
                if not job_dir.is_dir():
                    continue
                job_name = job_dir.name
                if job_name not in item_by_stem and _norm_name(job_name) not in item_by_norm:
                    continue
                if job_name in committed:
                    continue
                if _commit_job(job_name, job_dir):
                    processed += 1

        commit_poll = float((ctx.work_queue or {}).get("af3score_commit_poll_s", 5.0) or 5.0)
        commit_batch = int((ctx.work_queue or {}).get("af3score_commit_batch", 25) or 25)
        stop_evt = threading.Event()

        def _commit_loop() -> None:
            while not stop_evt.wait(commit_poll):
                try:
                    _scan_batch_outputs(max_per_pass=commit_batch)
                except Exception:
                    pass

        commit_thread = threading.Thread(target=_commit_loop, daemon=True)
        commit_thread.start()

        err: str | None = None
        if valid_items:
            try:
                run_command(
                    cmd,
                    env=os.environ.copy(),
                    cwd=str(batch_dir),
                    log_file=self.cfg.get("_log_file"),
                    verbose=bool(self.cfg.get("_verbose")),
                )
            except Exception as exc:
                err = str(exc)

        stop_evt.set()
        commit_thread.join(timeout=2.0)
        _scan_batch_outputs()

        # Fallback: split wrapper metrics.csv into per-item shards if any are missing.
        metrics_src = batch_dir / "metrics.csv"
        if metrics_src.exists():
            try:
                import pandas as pd

                df = pd.read_csv(metrics_src)
                for item in valid_items:
                    run_stem = str((item.payload or {}).get("run_stem") or item.id)
                    dst_metrics = metrics_items / f"{item.id}.csv"
                    if dst_metrics.exists():
                        # Repair known-bad promoted shards so retries can recover.
                        if is_valid_metrics_shard(dst_metrics, expected_desc=run_stem):
                            continue
                        try:
                            dst_metrics.unlink()
                        except Exception:
                            continue
                    run_stem_norm = run_stem.lower()
                    if run_stem_norm.endswith(".pdb"):
                        run_stem_norm = run_stem_norm[:-4]
                    mask = None
                    for col in ("description", "name", "model", "pdb_name"):
                        if col in df.columns:
                            series = df[col].astype(str)
                            series_norm = series.str.lower()
                            series_norm = series_norm.str.removesuffix(".pdb")
                            mask = series_norm == run_stem_norm
                            if mask.any():
                                break
                    if mask is not None and mask.any():
                        tmp = batch_dir / f".metrics_item.{item.id}.csv"
                        try:
                            df.loc[mask].to_csv(tmp, index=False)
                            promote_file_atomic(tmp, dst_metrics, allow_reuse=allow_reuse)
                        finally:
                            try:
                                tmp.unlink()
                            except Exception:
                                pass
                    elif len(valid_items) == 1:
                        tmp = batch_dir / f".metrics_item.{item.id}.csv"
                        try:
                            df.to_csv(tmp, index=False)
                            promote_file_atomic(tmp, dst_metrics, allow_reuse=allow_reuse)
                        finally:
                            try:
                                tmp.unlink()
                            except Exception:
                                pass
            except Exception:
                pass

        # Finalize CIF/PDB promotion for any remaining items.
        af3_dir = batch_dir / "af3score_outputs"
        if af3_dir.exists():
            for item in valid_items:
                run_stem = str((item.payload or {}).get("run_stem") or item.id)
                job_dir = af3_dir / run_stem
                if job_dir.exists():
                    source = _job_source(job_dir, run_stem)
                    if source is None:
                        continue
                    if not (cif_dir / f"{item.id}.cif").exists():
                        _promote_cif_for_source(source, item)
                    if not (pdbs_dir / f"{item.id}.pdb").exists():
                        _promote_pdb_for_source(source, item)

        all_done = all(self.item_done(ctx, item) for item in valid_items)
        if all_done:
            # Optional retention: copy heavy/raw runner outputs under output/_optional/... (never required for resume).
            try:
                keep_all = output_mode(ctx) == "full"
            except Exception:
                keep_all = False
            if keep_all or any(should_keep(ctx, k) for k in ["af3score_outputs", "af3_input_batch", "single_chain_cif", "json", "af3score_subprocess_logs"]):
                try:
                    opt_root = optional_dir(ctx) / out_dir.name
                    raw_root = opt_root / "raw"
                    logs_root = opt_root / "logs"
                    raw_keys = ["af3score_outputs", "af3_input_batch", "single_chain_cif", "json", "input_pdbs"]
                    log_keys = ["af3score_subprocess_logs"]
                    for key in raw_keys:
                        if not (keep_all or should_keep(ctx, key)):
                            continue
                        try:
                            promote_tree(batch_dir / key, raw_root / key, allow_reuse=True)
                        except Exception:
                            pass
                    for key in log_keys:
                        if not (keep_all or should_keep(ctx, key)):
                            continue
                        try:
                            promote_tree(batch_dir / key, logs_root / key, allow_reuse=True)
                        except Exception:
                            pass
                except Exception:
                    pass
            shutil.rmtree(batch_dir, ignore_errors=True)

        for item in valid_items:
            if item.id in forced_failed:
                results[item.id] = ("failed", forced_failed.get(item.id))
            elif self.item_done(ctx, item):
                results[item.id] = ("done", None)
            else:
                results[item.id] = ("failed", err or "missing output")
        return results

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        df = None
        metrics_items = out_dir / "metrics_items"
        if metrics_items.exists():
            metrics_paths = sorted(p for p in metrics_items.glob("*.csv"))
            if metrics_paths:
                try:
                    df = pd.concat([pd.read_csv(p) for p in metrics_paths], ignore_index=True)
                except Exception:
                    df = None
        if df is None:
            metrics_path = find_metrics_file(out_dir)
            if not metrics_path:
                return
            df = pd.read_csv(metrics_path)

        # Consolidated metrics.csv (atomic write)
        metrics_csv = out_dir / "metrics.csv"
        tmp_metrics = metrics_csv.parent / f"{metrics_csv.name}.tmp"
        df.to_csv(tmp_metrics, index=False)
        os.replace(tmp_metrics, metrics_csv)

        name_map = build_name_map(out_dir / "pdbs")

        def _get(row, *keys):
            for k in keys:
                if k in row and pd.notna(row[k]):
                    return row[k]
            return None

        def _get_ci(row, key: str):
            lower_map = {str(c).lower(): c for c in row.index}
            col = lower_map.get(key.lower())
            if col is not None and pd.notna(row[col]):
                return row[col]
            return None

        rows: list[dict[str, Any]] = []
        iptm_global_col: list[float | None] = []
        iptm_binder_target_col: list[float | None] = []
        protocol = ctx.input_data.get("protocol")
        is_antibody = protocol == "antibody"
        target_chain = "B"
        if is_antibody:
            target = ctx.input_data.get("target") or {}
            chains = target.get("chains")
            if isinstance(chains, list) and chains:
                target_chain = str(chains[0])
            elif isinstance(chains, str) and chains:
                target_chain = str(chains)
        for _, row in df.iterrows():
            desc = _get(row, "description", "name", "model", "pdb_name")
            desc = str(desc) if desc is not None else ""

            # Keep list lengths aligned with df rows for derived metrics_ppiflow.csv.
            iptm_global = _get(row, "iptm", "ipTM", "AF3Score_interchain_iptm", "AF3Score_chain_iptm")
            iptm_binder_target = None
            iptm = iptm_global
            if is_antibody:
                if desc:
                    chain_target_iptm = _get_ci(row, f"chain_{target_chain}_iptm")
                    if chain_target_iptm is not None:
                        iptm_binder_target = float(chain_target_iptm)
                        iptm = iptm_binder_target
                    else:
                        raise StepError(
                            "AF3Score metrics missing antibody binder-target iptm. "
                            "Require chain_<target>_iptm (e.g., chain_B_iptm)."
                        )
                else:
                    iptm = None
            ptm = _get(row, "ptm", "pTM", "ptm_A", "ptm_B", "AF3Score_chain_ptm")

            iptm_global_col.append(float(iptm_global) if iptm_global is not None else None)
            iptm_binder_target_col.append(
                float(iptm_binder_target) if iptm_binder_target is not None else None
            )

            if not desc:
                continue
            pdb_path = name_map.get(desc)
            rows.append({
                "design_id": extract_design_id(desc),
                "structure_id": structure_id_from_name(desc),
                "iptm": float(iptm) if iptm is not None else None,
                "ptm": float(ptm) if ptm is not None else None,
                "pdb_path": str(pdb_path) if pdb_path else None,
            })

        # Write derived metrics_ppiflow.csv (atomic write).
        df_ppi = df.copy()
        df_ppi["iptm_global"] = iptm_global_col
        df_ppi["iptm_binder_target"] = iptm_binder_target_col
        metrics_ppiflow = out_dir / "metrics_ppiflow.csv"
        tmp_ppi = metrics_ppiflow.parent / f"{metrics_ppiflow.name}.tmp"
        df_ppi.to_csv(tmp_ppi, index=False)
        os.replace(tmp_ppi, metrics_ppiflow)

        if rows:
            manifest_path = self.manifest_path(ctx)
            tmp_manifest = manifest_path.parent / f"{manifest_path.name}.tmp"
            write_csv(tmp_manifest, rows, ["design_id", "structure_id", "iptm", "ptm", "pdb_path"])
            os.replace(tmp_manifest, manifest_path)

    def outputs_complete(self, ctx: StepContext) -> bool:
        if not super().outputs_complete(ctx):
            return False
        out_dir = self._resolve_output_dir_path(ctx)
        if not (out_dir / "metrics.csv").exists():
            return False
        if not (out_dir / "metrics_ppiflow.csv").exists():
            return False
        if not (out_dir / "pdbs").exists() or not list((out_dir / "pdbs").glob("*.pdb")):
            return False
        if not (out_dir / "cif").exists() or not list((out_dir / "cif").glob("*.cif")):
            return False
        return True

    def _finalize_work_queue_outputs(self, ctx: StepContext, wq, *, items: list[Any], allow_failures: bool) -> None:
        # Strict finalize: do not write metadata unless derived outputs are successfully materialized.
        try:
            counts = wq.counts()
        except Exception:
            counts = None
        if counts and (counts.get("pending", 0) != 0 or counts.get("running", 0) != 0):
            return
        if not wq.acquire_leader():
            return
        try:
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

    def scan_done(self, ctx: StepContext) -> set[int]:
        done = super().scan_done(ctx)
        if done:
            return done
        out_dir = self.output_dir(ctx)
        metrics = out_dir / "metrics.csv"
        if metrics.exists():
            return set(range(self.expected_total(ctx)))
        return set()
