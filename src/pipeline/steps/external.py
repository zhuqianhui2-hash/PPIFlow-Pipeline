from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .base import Step, StepContext, StepError
from ..direct_legacy import compute_run_stems, promote_file, promote_tree
from ..io import collect_pdbs, is_ignored_path
from ..logging_utils import log_command_progress, run_command
from ..manifests import build_name_map, extract_design_id, find_metrics_file, structure_id_from_name, write_csv
from ..work_queue import WorkItem


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

    def _default_chain_list(self, ctx: StepContext) -> str:
        protocol = ctx.input_data.get("protocol")
        if protocol == "binder":
            return str(ctx.input_data.get("binder_chain") or "A")
        framework = ctx.input_data.get("framework") or {}
        return str(framework.get("heavy_chain") or "A")

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
                indices_raw = row.get("fixed_positions_indices") or row.get("fixed_positions") or ""
                indices = [int(x) for x in re.findall(r"\d+", str(indices_raw))]
                entry = {"chain": chain, "indices": indices}
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
            (seq_dir / name).write_text("\n".join(lines) + "\n")
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
            (seq_dir / name).write_text("\n".join(lines) + "\n")

    def _collect_pdbs(self, input_dir: Path) -> list[Path]:
        return collect_pdbs(input_dir)

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
        out_dir = self.output_dir(ctx)

        pdbs = self._collect_pdbs(input_path)
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
        return self._find_fasta(out_dir, stem) is not None

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        tools = ctx.input_data.get("tools") or {}
        out_dir = self.output_dir(ctx)
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
                "python",
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
            elif use_soluble_model:
                cmd.append("--use_soluble_model")
            else:
                raise StepError("Missing tools.mpnn_ckpt/abmpnn_ckpt for sequence design")
            if model_name:
                cmd.extend(["--model_name", model_name])
            elif protocol != "binder":
                cmd.extend(["--model_name", "abmpnn"])
            return cmd

        pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")
        if not run_stem:
            run_stem = pdb_path.stem
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
                    promote_file(fp, seq_dst / fp.name, allow_reuse=bool((ctx.work_queue or {}).get("allow_reuse", True)))
        shutil.rmtree(item_tmp, ignore_errors=True)

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
        use_soluble = bool(seq_cfg.get("use_soluble_ckpt"))
        if protocol == "binder" and use_soluble and self.cfg.get("name") == "seq2":
            ckpt = tools.get("mpnn_ckpt_soluble") or ckpt

        weight_dir, model_name = self._resolve_weights(ckpt)
        chain_arg = " ".join(str(chain_list).replace(",", " ").split())
        seed = str(int((ctx.state.get("runs") or [{}])[0].get("run_seed", 0) or 0))

        def build_base_cmd(out_folder: Path, num_seq_target: int, temp: float, use_soluble_model: bool) -> list[str]:
            cmd = [
                "python",
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
            elif use_soluble_model:
                cmd.append("--use_soluble_model")
            else:
                raise StepError("Missing tools.mpnn_ckpt/abmpnn_ckpt for sequence design")
            if model_name:
                cmd.extend(["--model_name", model_name])
            elif protocol != "binder":
                cmd.extend(["--model_name", "abmpnn"])
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
        cfg = self._flowpacker_config()
        input_pdb_dir = self._resolve_path(ctx, cfg.get("input_pdb_dir")) if cfg else None
        seq_fasta_dir = self._resolve_path(ctx, cfg.get("seq_fasta_dir")) if cfg else None
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
        flowpacker_out = out_dir / "flowpacker_outputs" / "run_1"
        if not flowpacker_out.exists():
            return False
        stem = str((item.payload or {}).get("run_stem") or (item.payload or {}).get("pdb_stem") or item.id)
        return any(fp.name.startswith(f"{stem}_") for fp in flowpacker_out.glob("*.pdb"))

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
            "python",
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
        flowpacker_out = out_dir / "flowpacker_outputs" / "run_1"
        after_pdbs = out_dir / "after_pdbs"
        promote_tree(item_out / "flowpacker_outputs" / "run_1", flowpacker_out, allow_reuse=allow_reuse)
        promote_tree(item_out / "after_pdbs", after_pdbs, allow_reuse=allow_reuse)
        shutil.rmtree(item_dir, ignore_errors=True)

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
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
        for sub in [out_dir / "flowpacker_outputs", out_dir / "after_pdbs", out_dir]:
            if sub.exists():
                for fp in sub.rglob("*.pdb"):
                    if is_ignored_path(fp):
                        continue
                    return set(range(self.expected_total(ctx)))
        return set()


class AF3ScoreStep(ExternalCommandStep):
    name = "af3score"
    stage = "score"
    supports_work_queue = True
    work_queue_mode = "items"

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
        cfg = self._af3score_config()
        input_pdb_dir = self._resolve_path(ctx, cfg.get("input_pdb_dir")) if cfg else None
        if not input_pdb_dir or not input_pdb_dir.exists():
            raise StepError("af3score input_pdb_dir missing or invalid")
        input_pdb_dir = self._resolve_input_dir(input_pdb_dir)

        pdbs = collect_pdbs(input_pdb_dir)
        if not pdbs:
            raise StepError(f"No PDBs found for af3score in {input_pdb_dir}")

        out_dir = self.output_dir(ctx)
        metrics_pdb_dir = out_dir / "metrics_pdbs"
        metrics_pdb_dir.mkdir(parents=True, exist_ok=True)
        run_stems = compute_run_stems(pdbs, input_pdb_dir)

        items: list[WorkItem] = []
        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        for pdb_path in pdbs:
            run_stem = run_stems[pdb_path]
            dst = metrics_pdb_dir / f"{run_stem}.pdb"
            if allow_reuse:
                try:
                    promote_file(pdb_path, dst, allow_reuse=allow_reuse)
                except Exception as exc:
                    print(
                        f"[af3score] WARN reuse collision at {dst}: {exc}",
                        file=sys.__stdout__,
                        flush=True,
                    )
            else:
                promote_file(pdb_path, dst, allow_reuse=allow_reuse)
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
        metrics_items = out_dir / "metrics_items"
        metrics_path = metrics_items / f"{item.id}.csv"
        return metrics_path.exists()

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

        out_dir = self.output_dir(ctx)
        item_dir = out_dir / ".tmp" / item.id
        item_dir.mkdir(parents=True, exist_ok=True)
        item_out = item_dir

        # Write per-item batch yaml for provenance.
        yaml_path = item_dir / "input.yaml"
        run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
        tmp_pdb = item_dir / f"{run_stem}.pdb"
        if not tmp_pdb.exists():
            try:
                os.link(pdb_path, tmp_pdb)
            except OSError:
                shutil.copy2(pdb_path, tmp_pdb)
        try:
            import yaml

            yaml_path.write_text(yaml.safe_dump({"input_pdb": str(tmp_pdb)}, sort_keys=False))
        except Exception:
            yaml_path.write_text(json.dumps({"input_pdb": str(tmp_pdb)}) + "\n")

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
            "python",
            str(script),
            "--input_pdb",
            str(tmp_pdb),
            "--output_dir",
            str(item_out),
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
        if cfg.get("export_pdb_dir") is not None:
            cmd.extend(["--export_pdb_dir", str(item_out / "pdbs")])
        if cfg.get("target_offsets_json"):
            offsets_path = self._resolve_path(ctx, cfg.get("target_offsets_json"))
            if offsets_path:
                cmd.extend(["--target_offsets_json", str(offsets_path)])
        if cfg.get("target_chain"):
            cmd.extend(["--target_chain", str(cfg.get("target_chain"))])

        run_command(
            cmd,
            env=os.environ.copy(),
            cwd=str(item_dir),
            log_file=self.cfg.get("_log_file"),
            verbose=bool(self.cfg.get("_verbose")),
        )

        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        for sub in [
            "af3score_outputs",
            "single_chain_cif",
            "json",
            "af3_input_batch",
            "pdbs",
            "af3score_subprocess_logs",
        ]:
            promote_tree(item_out / sub, out_dir / sub, allow_reuse=allow_reuse)
        metrics_items = out_dir / "metrics_items"
        metrics_items.mkdir(parents=True, exist_ok=True)
        metrics_src = item_out / "metrics.csv"
        if metrics_src.exists():
            promote_file(metrics_src, metrics_items / f"{item.id}.csv", allow_reuse=allow_reuse)
        shutil.rmtree(item_dir, ignore_errors=True)

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        df = None
        metrics_items = out_dir / "metrics_items"
        if metrics_items.exists():
            metrics_paths = sorted(p for p in metrics_items.glob("*.csv"))
            if metrics_paths:
                try:
                    df = pd.concat([pd.read_csv(p) for p in metrics_paths], ignore_index=True)
                    try:
                        df.to_csv(out_dir / "metrics.csv", index=False)
                    except Exception:
                        pass
                except Exception:
                    df = None

        if df is None:
            metrics_path = find_metrics_file(out_dir)
            if not metrics_path:
                return
            df = pd.read_csv(metrics_path)
        # Prefer the FlowPacker inputs captured for metrics to avoid AF3Score CIF->PDB issues.
        metrics_pdb_dir = out_dir / "metrics_pdbs"
        step_label = str(self.cfg.get("name") or self.name)
        try:
            name_map = build_name_map(metrics_pdb_dir)
        except Exception as exc:
            print(
                f"[{step_label}] WARN failed to scan metrics_pdbs at {metrics_pdb_dir}: {exc}",
                flush=True,
            )
            name_map = {}
        if not name_map:
            print(
                f"[{step_label}] WARN no PDBs found in {metrics_pdb_dir}; "
                "AF3Score rows will have missing pdb_path values.",
                flush=True,
            )

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
        missing_pdbs: list[str] = []
        for _, row in df.iterrows():
            desc = _get(row, "description", "name", "model", "pdb_name")
            desc = str(desc) if desc is not None else ""
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
            pdb_path = name_map.get(desc)
            if desc and pdb_path is None:
                missing_pdbs.append(desc)
            iptm_global_col.append(float(iptm_global) if iptm_global is not None else None)
            iptm_binder_target_col.append(
                float(iptm_binder_target) if iptm_binder_target is not None else None
            )
            rows.append({
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
                f"[{step_label}] WARN missing {len(missing_pdbs)} PDBs in {metrics_pdb_dir} "
                f"for AF3Score rows (examples: {sample}).",
                flush=True,
            )

        # Write derived metrics with binder-target ipTM if needed
        df = df.copy()
        df["iptm_global"] = iptm_global_col
        df["iptm_binder_target"] = iptm_binder_target_col
        metrics_ppiflow = out_dir / "metrics_ppiflow.csv"
        try:
            df.to_csv(metrics_ppiflow, index=False)
        except Exception:
            pass

        if not rows:
            return

        filters = (ctx.input_data.get("filters") or {}).get("af3score") or {}
        if self.cfg.get("name") == "af3score2":
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
        if self.cfg.get("name") == "af3score2":
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

        # Write filtered PDBs for downstream steps
        filtered_dir = out_dir / "filtered_pdbs"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if not r.get("passed_filter"):
                continue
            if top_k is not None and not r.get("passed_top_k"):
                continue
            pdb_path = r.get("pdb_path")
            if not pdb_path:
                continue
            src = Path(str(pdb_path))
            if not src.exists():
                continue
            dst = filtered_dir / src.name
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except Exception:
                try:
                    dst.write_bytes(src.read_bytes())
                except Exception:
                    pass

        write_csv(
            self.manifest_path(ctx),
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

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        metrics_path = find_metrics_file(out_dir)
        if not metrics_path:
            return
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            return
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
            if not desc:
                continue
            iptm = _get(row, "iptm", "ipTM", "AF3Score_interchain_iptm", "AF3Score_chain_iptm")
            ptm = _get(row, "ptm", "pTM", "ptm_A", "ptm_B", "AF3Score_chain_ptm")
            if is_antibody:
                chain_target_iptm = _get_ci(row, f"chain_{target_chain}_iptm")
                if chain_target_iptm is not None:
                    iptm = float(chain_target_iptm)
                else:
                    raise StepError(
                        "AF3Score metrics missing antibody binder-target iptm. "
                        "Require chain_<target>_iptm (e.g., chain_B_iptm)."
                    )
            pdb_path = name_map.get(desc)
            rows.append({
                "design_id": extract_design_id(desc),
                "structure_id": structure_id_from_name(desc),
                "iptm": float(iptm) if iptm is not None else None,
                "ptm": float(ptm) if ptm is not None else None,
                "pdb_path": str(pdb_path) if pdb_path else None,
            })

        if rows:
            write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "iptm", "ptm", "pdb_path"])

    def run_full(self, ctx: StepContext) -> None:
        cmd = self.cfg.get("command")
        if not cmd:
            return
        return super().run_full(ctx)

    def scan_done(self, ctx: StepContext) -> set[int]:
        done = super().scan_done(ctx)
        if done:
            return done
        out_dir = self.output_dir(ctx)
        metrics = out_dir / "metrics.csv"
        if metrics.exists():
            return set(range(self.expected_total(ctx)))
        return set()
