from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from .base import Step, StepContext, StepError
from ..direct_legacy import compute_run_stems, promote_file, promote_tree
from ..io import collect_pdbs, is_ignored_path
from ..logging_utils import log_command_progress, run_command
from ..manifests import extract_design_id, structure_id_from_name, write_csv
from ..output_policy import is_minimal, optional_dir, step_scratch_dir
from ..work_queue import WorkItem


def _pipeline_root() -> Path:
    # steps/rosetta_steps.py -> steps -> pipeline
    return Path(__file__).resolve().parents[1]


def _find_rosetta_resource(rel: str) -> Path:
    base = _pipeline_root() / "rosetta"
    path = base / rel
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing rosetta resource: {path}")


def _parse_pdb_chains(pdb_path: Path) -> dict[str, tuple[int, int]]:
    chains: dict[str, tuple[int, int]] = {}
    with pdb_path.open("r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21]
            try:
                res_seq = int(line[22:26].strip())
            except ValueError:
                continue
            if chain_id not in chains:
                chains[chain_id] = (res_seq, res_seq)
            else:
                start, end = chains[chain_id]
                chains[chain_id] = (start, res_seq)
    return chains


def _build_resnums(chains: dict[str, tuple[int, int]]) -> str:
    parts = []
    for chain, (start, end) in chains.items():
        parts.append(f"{start}{chain}-{end}{chain}")
    return ",".join(parts)


def _write_update_xml(template_text: str, pdb_path: Path, xml_path: Path) -> None:
    chains = _parse_pdb_chains(pdb_path)
    resnums = _build_resnums(chains)
    new_content = re.sub(r'resnums="[^"]+"', f'resnums="{resnums}"', template_text)
    xml_path.write_text(new_content)


def _resolve_rosetta_cmd(rosetta_bin: str) -> list[str]:
    env_name = os.environ.get("ROSETTA_ENV") or os.environ.get("ROSETTA_CONDA_ENV")
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if env_name and conda_exe:
        return [conda_exe, "run", "-n", env_name, rosetta_bin]
    return [rosetta_bin]


def _resolve_binder_chain(input_data: dict[str, Any]) -> str:
    protocol = input_data.get("protocol")
    if protocol in {"antibody", "vhh"}:
        framework = input_data.get("framework") or {}
        return str(framework.get("heavy_chain") or "A")
    return str(input_data.get("binder_chain") or "A")


def _resolve_target_chains(input_data: dict[str, Any]) -> list[str]:
    target = input_data.get("target") or {}
    chains = target.get("chains") or []
    if isinstance(chains, str):
        chains = [c.strip() for c in chains.split(",") if c.strip()]
    chains = [str(c) for c in chains]
    if not chains:
        raise StepError("target.chains required for rosetta_interface")
    return chains


def _run(cmd: list[str], cwd: Path | None, stdout_path: Path | None, env: dict[str, str], verbose: bool) -> None:
    log_prefix = env.get("PPIFLOW_LOG_PREFIX")
    worker_log = env.get("PPIFLOW_LOG_PATH")
    if stdout_path is None or not (log_prefix and worker_log):
        # Default path: single log file (or console) via run_command.
        run_command(cmd, cwd=cwd, env=env, log_file=stdout_path, verbose=verbose)
        return

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    Path(worker_log).parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a") as out_handle, Path(worker_log).open("a") as worker_handle:
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
            out_handle.write(line)
            worker_handle.write(f"[{log_prefix}] {line}")
            if verbose:
                sys.stdout.write(f"[{log_prefix}] {line}")
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, list(cmd))


def _run_job(
    args: tuple[str, list[str], Path, Path, dict[str, str], bool]
) -> tuple[str, bool, float, str | None, Path]:
    name, cmd, cwd, out_path, env, verbose = args
    start = time.time()
    try:
        _run(cmd, cwd, out_path, env, verbose)
        return name, True, time.time() - start, None, out_path
    except Exception as exc:
        return name, False, time.time() - start, str(exc), out_path


class RosettaInterfaceStep(Step):
    name = "rosetta_interface"
    stage = "rosetta"
    supports_indices = False
    supports_work_queue = True
    work_queue_mode = "items"

    def expected_total(self, ctx: StepContext) -> int:
        return 1

    def _collect_input_pdbs(self, ctx: StepContext) -> list[Path]:
        input_dir = self.cfg.get("input_dir")
        if not input_dir:
            raise StepError("rosetta_interface missing input_dir")
        input_path = Path(input_dir)
        if not input_path.is_absolute():
            input_path = ctx.out_dir / input_path
        if not input_path.exists():
            raise StepError(f"rosetta_interface input_dir not found: {input_path}")
        pdbs = collect_pdbs(input_path)
        if not pdbs:
            raise StepError("No PDBs found for rosetta_interface")
        return pdbs

    def scan_done(self, ctx: StepContext) -> set[int]:
        out_dir = self.output_dir(ctx)
        if (out_dir / "residue_energy.csv").exists():
            return {0}
        return set()

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        pdbs = self._collect_input_pdbs(ctx)
        items: list[WorkItem] = []
        input_root = Path(self.cfg.get("input_dir") or "")
        if not input_root.is_absolute():
            input_root = ctx.out_dir / input_root
        run_stems = compute_run_stems(pdbs, input_root)
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
        items_dir = out_dir / "residue_energy_items"
        return (items_dir / f"{item.id}.csv").exists()

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        tools = ctx.input_data.get("tools") or {}
        rosetta_bin = tools.get("rosetta_bin") or os.environ.get("ROSETTA_BIN")
        if not rosetta_bin:
            raise StepError("rosetta_bin is required for rosetta_interface")
        rosetta_db = tools.get("rosetta_db") or os.environ.get("ROSETTA_DB")

        template_path = _find_rosetta_resource("native.xml")
        template_text = template_path.read_text()

        out_dir = self.output_dir(ctx)
        item_dir = out_dir / ".tmp" / item.id
        item_dir.mkdir(parents=True, exist_ok=True)
        item_out = item_dir

        input_dir = item_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
        input_pdb = input_dir / f"{run_stem}.pdb"
        if not input_pdb.exists():
            shutil.copy2(pdb_path, input_pdb)

        step_name = str(self.cfg.get("name") or self.name)
        output_enabled = isinstance(ctx.input_data.get("output"), dict)
        if output_enabled:
            job_root = step_scratch_dir(ctx, step_name) / item.id / "rosetta_jobs"
        else:
            job_root = item_out / "rosetta_jobs"
        job_root.mkdir(parents=True, exist_ok=True)
        name = run_stem
        job_dir = job_root / name
        out_dir_job = job_dir / "out"
        out_dir_job.mkdir(parents=True, exist_ok=True)
        job_pdb = job_dir / f"{name}.pdb"
        if not job_pdb.exists():
            shutil.copy2(input_pdb, job_pdb)
        xml_path = job_dir / "update.xml"
        _write_update_xml(template_text, job_pdb, xml_path)
        out_path = out_dir_job / f"{name}.out"
        if not out_path.exists():
            cmd = _resolve_rosetta_cmd(str(rosetta_bin)) + [
                "-parser:protocol",
                str(xml_path),
                "-s",
                str(job_pdb),
                "-overwrite",
                "-ignore_zero_occupancy",
                "false",
            ]
            if rosetta_db:
                cmd.extend(["-database", str(rosetta_db)])
            _run(cmd, job_dir, out_path, os.environ.copy(), bool(self.cfg.get("_verbose")))

        script = _find_rosetta_resource("get_interface_energy.py")
        binder_id = _resolve_binder_chain(ctx.input_data)
        target_ids = _resolve_target_chains(ctx.input_data)
        interface_dist = float((ctx.input_data.get("filters") or {}).get("rosetta", {}).get("interface_distance") or 12.0)
        target_id_arg = ",".join(target_ids)
        cmd = [
            sys.executable,
            str(script),
            "--input_pdbdir",
            str(input_dir),
            "--rosetta_dir",
            str(job_root),
            "--binder_id",
            binder_id,
            "--target_id",
            target_id_arg,
            "--output_dir",
            str(item_out),
            "--interface_dist",
            str(interface_dist),
        ]
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        run_command(
            cmd,
            env=env,
            cwd=str(item_dir),
            log_file=self.cfg.get("_log_file"),
            verbose=bool(self.cfg.get("_verbose")),
        )

        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        if output_enabled and not is_minimal(ctx):
            promote_tree(job_root, optional_dir(ctx) / "rosetta_interface" / "rosetta_jobs", allow_reuse=allow_reuse)
        residue_items = out_dir / "residue_energy_items"
        residue_items.mkdir(parents=True, exist_ok=True)
        residue_src = item_out / "residue_energy.csv"
        if residue_src.exists():
            # Rewrite pdbpath to point at the original input PDB (not .tmp) for downstream steps.
            try:
                import pandas as pd

                df = pd.read_csv(residue_src)
                if "pdbpath" in df.columns:
                    df["pdbpath"] = str(pdb_path)
                df.to_csv(residue_src, index=False)
            except Exception:
                pass
            promote_file(residue_src, residue_items / f"{item.id}.csv", allow_reuse=allow_reuse)
        shutil.rmtree(item_dir, ignore_errors=True)
        if output_enabled:
            shutil.rmtree(job_root.parent, ignore_errors=True)

    def run_full(self, ctx: StepContext) -> None:
        input_dir = self.cfg.get("input_dir")
        if not input_dir:
            raise StepError("rosetta_interface missing input_dir")
        input_path = Path(input_dir)
        if not input_path.is_absolute():
            input_path = ctx.out_dir / input_path
        if not input_path.exists():
            raise StepError(f"rosetta_interface input_dir not found: {input_path}")

        tools = ctx.input_data.get("tools") or {}
        rosetta_bin = tools.get("rosetta_bin") or os.environ.get("ROSETTA_BIN")
        if not rosetta_bin:
            raise StepError("rosetta_bin is required for rosetta_interface")
        rosetta_db = tools.get("rosetta_db") or os.environ.get("ROSETTA_DB")

        template_path = _find_rosetta_resource("native.xml")
        template_text = template_path.read_text()

        pdbs = collect_pdbs(input_path)

        if not pdbs:
            raise StepError("No PDBs found for rosetta_interface")
        run_stems = compute_run_stems(pdbs, input_path)

        out_dir = self.output_dir(ctx)
        step_name = str(self.cfg.get("name") or self.name)
        output_enabled = isinstance(ctx.input_data.get("output"), dict)
        if output_enabled:
            rosetta_root = step_scratch_dir(ctx, step_name) / "rosetta_jobs"
        else:
            rosetta_root = out_dir / "rosetta_jobs"
        rosetta_root.mkdir(parents=True, exist_ok=True)

        jobs: list[tuple[list[str], Path, Path]] = []
        for pdb_path in pdbs:
            name = run_stems[pdb_path]
            job_dir = rosetta_root / name
            out_dir_job = job_dir / "out"
            out_dir_job.mkdir(parents=True, exist_ok=True)
            job_pdb = job_dir / f"{name}.pdb"
            if not job_pdb.exists():
                shutil.copy2(pdb_path, job_pdb)
            xml_path = job_dir / "update.xml"
            _write_update_xml(template_text, job_pdb, xml_path)
            out_path = out_dir_job / f"{name}.out"
            if out_path.exists():
                continue
            cmd = _resolve_rosetta_cmd(str(rosetta_bin)) + [
                "-parser:protocol",
                str(xml_path),
                "-s",
                str(job_pdb),
                "-overwrite",
                "-ignore_zero_occupancy",
                "false",
            ]
            if rosetta_db:
                cmd.extend(["-database", str(rosetta_db)])
            jobs.append((cmd, job_dir, out_path))

        if jobs:
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            workers = min(len(jobs), os.cpu_count() or 1)
            total = len(jobs)
            failures: list[str] = []
            allow_failures = bool((ctx.input_data.get("options") or {}).get("continue_on_item_error"))
            with Pool(processes=workers) as pool:
                args_iter = [
                    (job_dir.name, cmd, job_dir, out, env, bool(self.cfg.get("_verbose")))
                    for cmd, job_dir, out in jobs
                ]
                for idx, (name, ok, elapsed, err, out_path) in enumerate(
                    pool.imap_unordered(_run_job, args_iter), start=1
                ):
                    status = "OK" if ok else "FAILED"
                    log_command_progress(
                        str(self.cfg.get("name") or self.name),
                        idx,
                        total,
                        item=name,
                        status=status,
                        elapsed=elapsed,
                        log_file=out_path,
                        extra=None,
                    )
                    if not ok and err:
                        failures.append(f"{name}: {err}")
            if failures:
                (out_dir / "rosetta_interface_failures.txt").write_text("\n".join(failures) + "\n")
                if not allow_failures or len(failures) >= total:
                    sample = "; ".join(failures[:3])
                    raise StepError(f"rosetta_interface failed for {len(failures)} jobs (e.g., {sample})")

        # Run parser to generate residue_energy.csv
        script = _find_rosetta_resource("get_interface_energy.py")
        binder_id = _resolve_binder_chain(ctx.input_data)
        target_ids = _resolve_target_chains(ctx.input_data)
        interface_dist = float((ctx.input_data.get("filters") or {}).get("rosetta", {}).get("interface_distance") or 12.0)
        target_id_arg = ",".join(target_ids)
        cmd = [
            sys.executable,
            str(script),
            "--input_pdbdir",
            str(input_path),
            "--rosetta_dir",
            str(rosetta_root),
            "--binder_id",
            binder_id,
            "--target_id",
            target_id_arg,
            "--output_dir",
            str(out_dir),
            "--interface_dist",
            str(interface_dist),
        ]
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
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
                str(self.cfg.get("name") or self.name),
                1,
                1,
                item="parse_interface",
                status=status,
                elapsed=time.time() - start,
                log_file=self.cfg.get("_log_file"),
            )

        # Optional interface score summary for ranking
        residue_csv = out_dir / "residue_energy.csv"
        if residue_csv.exists():
            try:
                import ast
                import pandas as pd

                df = pd.read_csv(residue_csv)
                rows = []
                for _, row in df.iterrows():
                    name = str(row.get("pdbname") or Path(str(row.get("pdbpath", ""))).stem)
                    try:
                        energy_dict = ast.literal_eval(row.get("binder_energy") or "{}")
                    except Exception:
                        energy_dict = {}
                    interface_score = sum(float(v) for v in (energy_dict or {}).values())
                    rows.append({"pdb_name": name, "interface_score": interface_score})
                if rows:
                    write_csv(out_dir / "interface_scores.csv", rows, ["pdb_name", "interface_score"])
            except Exception:
                pass

        if output_enabled and not is_minimal(ctx):
            try:
                promote_tree(rosetta_root, optional_dir(ctx) / "rosetta_interface" / "rosetta_jobs", allow_reuse=True)
            except Exception:
                pass
        if output_enabled:
            shutil.rmtree(rosetta_root, ignore_errors=True)

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        residue_items = out_dir / "residue_energy_items"
        if residue_items.exists():
            residue_rows = []
            for src_csv in sorted(residue_items.glob("*.csv")):
                try:
                    import pandas as pd

                    df_item = pd.read_csv(src_csv)
                    residue_rows.append(df_item)
                except Exception:
                    pass
            if residue_rows:
                try:
                    import pandas as pd

                    df_all = pd.concat(residue_rows, ignore_index=True)
                    df_all.to_csv(out_dir / "residue_energy.csv", index=False)
                    # Optional interface score summary for ranking
                    rows = []
                    for _, row in df_all.iterrows():
                        name = str(row.get("pdbname") or Path(str(row.get("pdbpath", ""))).stem)
                        try:
                            import ast

                            energy_dict = ast.literal_eval(row.get("binder_energy") or "{}")
                        except Exception:
                            energy_dict = {}
                        interface_score = sum(float(v) for v in (energy_dict or {}).values())
                        rows.append({"pdb_name": name, "interface_score": interface_score})
                    if rows:
                        write_csv(out_dir / "interface_scores.csv", rows, ["pdb_name", "interface_score"])
                except Exception:
                    pass
        residue_csv = out_dir / "residue_energy.csv"
        if not residue_csv.exists():
            return
        try:
            import pandas as pd

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
        if rows:
            write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "residue_energy_csv"])


class RosettaRelaxStep(Step):
    name = "relax"
    stage = "rosetta"
    supports_indices = False
    supports_work_queue = True
    work_queue_mode = "items"

    def expected_total(self, ctx: StepContext) -> int:
        return 1

    def scan_done(self, ctx: StepContext) -> set[int]:
        out_dir = self.output_dir(ctx)
        for fp in out_dir.rglob("*.pdb"):
            if is_ignored_path(fp):
                continue
            return {0}
        return set()

    def _collect_input_pdbs(self, ctx: StepContext) -> list[Path]:
        input_dir = self.cfg.get("input_dir")
        if not input_dir:
            raise StepError("relax missing input_dir")
        input_path = Path(input_dir)
        if not input_path.is_absolute():
            input_path = ctx.out_dir / input_path
        if not input_path.exists():
            raise StepError(f"relax input_dir not found: {input_path}")
        pdbs = collect_pdbs(input_path)
        if not pdbs:
            raise StepError("No PDBs found for relax")
        return pdbs

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        pdbs = self._collect_input_pdbs(ctx)
        items: list[WorkItem] = []
        input_root = Path(self.cfg.get("input_dir") or "")
        if not input_root.is_absolute():
            input_root = ctx.out_dir / input_root
        run_stems = compute_run_stems(pdbs, input_root)
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
        stem = str((item.payload or {}).get("run_stem") or (item.payload or {}).get("pdb_name") or item.id)
        return (out_dir / f"{stem}.pdb").exists()

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        pdb_path = Path(str((item.payload or {}).get("pdb_path") or ""))
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        tools = ctx.input_data.get("tools") or {}
        rosetta_bin = tools.get("rosetta_bin") or os.environ.get("ROSETTA_BIN")
        if not rosetta_bin:
            raise StepError("rosetta_bin is required for relax")
        rosetta_db = tools.get("rosetta_db") or os.environ.get("ROSETTA_DB")

        template_path = _find_rosetta_resource("native.xml")
        template_text = template_path.read_text()

        out_dir = self.output_dir(ctx)
        item_dir = out_dir / ".tmp" / item.id
        item_dir.mkdir(parents=True, exist_ok=True)
        item_out = item_dir

        input_dir = item_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        run_stem = str((item.payload or {}).get("run_stem") or pdb_path.stem)
        input_pdb = input_dir / f"{run_stem}.pdb"
        if not input_pdb.exists():
            shutil.copy2(pdb_path, input_pdb)

        step_name = str(self.cfg.get("name") or self.name)
        output_enabled = isinstance(ctx.input_data.get("output"), dict)
        if output_enabled:
            job_root = step_scratch_dir(ctx, step_name) / item.id / "rosetta_jobs"
        else:
            job_root = item_out / "rosetta_jobs"
        job_root.mkdir(parents=True, exist_ok=True)
        name = run_stem
        job_dir = job_root / name
        out_dir_job = job_dir / "out"
        out_dir_job.mkdir(parents=True, exist_ok=True)
        job_pdb = job_dir / f"{name}.pdb"
        if not job_pdb.exists():
            shutil.copy2(input_pdb, job_pdb)
        xml_path = job_dir / "update.xml"
        _write_update_xml(template_text, job_pdb, xml_path)
        log_path = out_dir_job / f"{name}.out"
        if not log_path.exists():
            cmd = _resolve_rosetta_cmd(str(rosetta_bin)) + [
                "-parser:protocol",
                str(xml_path),
                "-s",
                str(job_pdb),
                "-overwrite",
                "-ignore_zero_occupancy",
                "false",
            ]
            if rosetta_db:
                cmd.extend(["-database", str(rosetta_db)])
            _run(cmd, job_dir, log_path, os.environ.copy(), bool(self.cfg.get("_verbose")))

        # Collect relaxed PDB (rosetta_scripts writes *_0001.pdb by default)
        relaxed = None
        for pdb in sorted(job_dir.glob("*.pdb")):
            if pdb.name.endswith("_0001.pdb") or pdb.name.endswith("_0001.pdb.gz"):
                relaxed = pdb
                break
        if relaxed is None:
            raise StepError(f"Relax output missing for {name}")
        target = out_dir / f"{name}.pdb"
        allow_reuse = bool((ctx.work_queue or {}).get("allow_reuse", True))
        promote_file(relaxed, target, allow_reuse=allow_reuse)
        if output_enabled and not is_minimal(ctx):
            promote_tree(job_root, optional_dir(ctx) / "relax" / "rosetta_jobs", allow_reuse=allow_reuse)
        shutil.rmtree(item_dir, ignore_errors=True)
        if output_enabled:
            shutil.rmtree(job_root.parent, ignore_errors=True)

    def run_full(self, ctx: StepContext) -> None:
        input_dir = self.cfg.get("input_dir")
        if not input_dir:
            raise StepError("relax missing input_dir")
        input_path = Path(input_dir)
        if not input_path.is_absolute():
            input_path = ctx.out_dir / input_path
        if not input_path.exists():
            raise StepError(f"relax input_dir not found: {input_path}")

        tools = ctx.input_data.get("tools") or {}
        rosetta_bin = tools.get("rosetta_bin") or os.environ.get("ROSETTA_BIN")
        if not rosetta_bin:
            raise StepError("rosetta_bin is required for relax")
        rosetta_db = tools.get("rosetta_db") or os.environ.get("ROSETTA_DB")

        template_path = _find_rosetta_resource("native.xml")
        template_text = template_path.read_text()

        pdbs = collect_pdbs(input_path)

        if not pdbs:
            raise StepError("No PDBs found for relax")
        run_stems = compute_run_stems(pdbs, input_path)

        out_dir = self.output_dir(ctx)
        step_name = str(self.cfg.get("name") or self.name)
        output_enabled = isinstance(ctx.input_data.get("output"), dict)
        if output_enabled:
            rosetta_root = step_scratch_dir(ctx, step_name) / "rosetta_jobs"
        else:
            rosetta_root = out_dir / "rosetta_jobs"
        rosetta_root.mkdir(parents=True, exist_ok=True)

        jobs: list[tuple[list[str], Path, Path, Path]] = []
        for pdb_path in pdbs:
            name = run_stems[pdb_path]
            job_dir = rosetta_root / name
            out_dir_job = job_dir / "out"
            out_dir_job.mkdir(parents=True, exist_ok=True)
            job_pdb = job_dir / f"{name}.pdb"
            if not job_pdb.exists():
                shutil.copy2(pdb_path, job_pdb)
            xml_path = job_dir / "update.xml"
            _write_update_xml(template_text, job_pdb, xml_path)
            log_path = out_dir_job / f"{name}.out"
            if log_path.exists():
                continue
            cmd = _resolve_rosetta_cmd(str(rosetta_bin)) + [
                "-parser:protocol",
                str(xml_path),
                "-s",
                str(job_pdb),
                "-overwrite",
                "-ignore_zero_occupancy",
                "false",
            ]
            if rosetta_db:
                cmd.extend(["-database", str(rosetta_db)])
            jobs.append((cmd, job_dir, log_path, job_pdb))

        if jobs:
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            workers = min(len(jobs), os.cpu_count() or 1)
            total = len(jobs)
            failures: list[str] = []
            allow_failures = bool((ctx.input_data.get("options") or {}).get("continue_on_item_error"))
            with Pool(processes=workers) as pool:
                args_iter = [
                    (job_dir.name, cmd, job_dir, out, env, bool(self.cfg.get("_verbose")))
                    for cmd, job_dir, out, _ in jobs
                ]
                for idx, (name, ok, elapsed, err, out_path) in enumerate(
                    pool.imap_unordered(_run_job, args_iter), start=1
                ):
                    status = "OK" if ok else "FAILED"
                    log_command_progress(
                        str(self.cfg.get("name") or self.name),
                        idx,
                        total,
                        item=name,
                        status=status,
                        elapsed=elapsed,
                        log_file=out_path,
                        extra=None,
                    )
                    if not ok and err:
                        failures.append(f"{name}: {err}")
            if failures:
                (out_dir / "relax_failures.txt").write_text("\n".join(failures) + "\n")
                if not allow_failures or len(failures) >= total:
                    sample = "; ".join(failures[:3])
                    raise StepError(f"relax failed for {len(failures)} jobs (e.g., {sample})")

        # Collect relaxed PDBs (rosetta_scripts writes *_0001.pdb by default)
        for job in rosetta_root.iterdir():
            if not job.is_dir():
                continue
            pdbs_out = sorted(job.glob("*.pdb"))
            for pdb in pdbs_out:
                if pdb.name.endswith("_0001.pdb") or pdb.name.endswith("_0001.pdb.gz"):
                    target = out_dir / pdb.name.replace("_0001", "")
                    if not target.exists():
                        shutil.copy2(pdb, target)
                    break
        if output_enabled and not is_minimal(ctx):
            try:
                promote_tree(rosetta_root, optional_dir(ctx) / "relax" / "rosetta_jobs", allow_reuse=True)
            except Exception:
                pass
        if output_enabled:
            shutil.rmtree(rosetta_root, ignore_errors=True)

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
        if rows:
            write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "pdb_path"])
