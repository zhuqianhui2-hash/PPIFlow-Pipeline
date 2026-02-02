#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path


def _resolve_input_dir(input_dir: Path) -> Path:
    if (input_dir / "run_1").exists():
        return input_dir / "run_1"
    return input_dir


def _bucket_from_name(name: str, default: int = 256) -> int:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))
    return default


def _progress(idx: int, total: int, phase: str, item: str, status: str, elapsed: float) -> None:
    print(
        f"[af3score] {idx}/{total} {status} elapsed={elapsed:.2f}s phase={phase} item={item}",
        flush=True,
    )


def _quiet_enabled() -> bool:
    value = os.environ.get("PPIFLOW_AF3SCORE_QUIET", "")
    return value.strip().lower() in {"1", "true", "yes"}


def _run_cmd(cmd: list[str], *, env: dict, log_path: Path | None = None, cwd: str | None = None) -> None:
    if log_path is None:
        subprocess.check_call(cmd, env=env, cwd=cwd)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        subprocess.check_call(cmd, env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)


def _write_progress(
    output_dir: Path,
    produced: int,
    expected: int,
    *,
    phase: str,
    item: str | None = None,
    status: str = "running",
) -> None:
    payload = {
        "expected_total": max(int(expected), 0),
        "produced_total": max(int(produced), 0),
        "status": status,
        "phase": phase,
        "item": item,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        path = output_dir / "progress.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")))
        tmp.replace(path)
    except Exception:
        pass


def _resolve_python_cmd() -> list[str]:
    override = os.environ.get("AF3SCORE_PYTHON")
    if override:
        return shlex.split(override)
    env_name = os.environ.get("AF3SCORE_ENV", "ppiflow-af3score")
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_exe and env_name:
        return [conda_exe, "run", "-n", env_name, "python"]
    # Fallback to the bundled AF3Score env if available (default install layout).
    try:
        repo_root = Path(__file__).resolve().parents[1]
        local_py = repo_root / ".miniforge3" / "envs" / "ppiflow-af3score" / "bin" / "python"
        if local_py.exists():
            return [str(local_py)]
    except Exception:
        pass
    return ["python"]


def _parse_seed_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    seeds: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            step = 1 if end >= start else -1
            seeds.extend(list(range(start, end + step, step)))
        else:
            try:
                seeds.append(int(token))
            except ValueError:
                continue
    return seeds or None


def _strip_templates_and_set_seeds(json_path: Path, *, no_templates: bool, model_seeds: list[int] | None) -> None:
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return
    if no_templates:
        for entry in data.get("sequences", []):
            protein = entry.get("protein") if isinstance(entry, dict) else None
            if isinstance(protein, dict) and "templates" in protein:
                protein["templates"] = []
    if model_seeds:
        data["modelSeeds"] = model_seeds
    try:
        json_path.write_text(json.dumps(data, indent=2))
    except Exception:
        return


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


def _prepare_metrics_pdb_dir(input_pdb_dir: Path, output_dir_af3score: Path, work_dir: Path) -> Path:
    metrics_dir = work_dir / "metrics_pdbs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    input_map = {fp.stem.lower(): fp for fp in input_pdb_dir.rglob("*.pdb")}
    for job_dir in sorted(output_dir_af3score.iterdir()):
        if not job_dir.is_dir():
            continue
        name = job_dir.name
        dst = metrics_dir / f"{name}.pdb"
        if dst.exists():
            continue
        src = input_pdb_dir / f"{name}.pdb"
        if not src.exists():
            src = input_map.get(name.lower())
        if not src or not src.exists():
            continue
        try:
            os.link(src, dst)
        except Exception:
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass
    return metrics_dir


def _ensure_seed10_alias(output_dir_af3score: Path) -> None:
    for job_dir in sorted(output_dir_af3score.iterdir()):
        if not job_dir.is_dir():
            continue
        seed10 = job_dir / "seed-10_sample-0"
        if seed10.exists():
            continue
        seed_dirs = sorted([p for p in job_dir.glob("seed-*_sample-0") if p.is_dir()])
        if not seed_dirs:
            continue
        target = seed_dirs[0]
        try:
            os.symlink(target.name, seed10)
        except FileExistsError:
            continue
        except Exception:
            try:
                shutil.copytree(target, seed10)
            except Exception:
                continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdb_dir", default=None)
    parser.add_argument("--input_pdb", default=None, help="Single PDB input (per-item mode)")
    parser.add_argument("--batch_yaml", default=None, help="YAML with input_pdb or input_pdb_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--af3score_repo", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--db_dir", default=None)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--bucket_default", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--model_seeds", type=str, default=None)
    parser.add_argument("--no_templates", action="store_true")
    parser.add_argument("--write_cif_model", type=str, default="false")
    parser.add_argument("--write_best_model_root", action="store_true")
    parser.add_argument("--write_ranking_scores_csv", action="store_true")
    parser.add_argument("--export_pdb_dir", type=str, default=None)
    parser.add_argument("--target_offsets_json", type=str, default=None)
    parser.add_argument("--target_chain", type=str, default="B")
    args = parser.parse_args()

    if args.batch_yaml:
        try:
            import yaml

            payload = yaml.safe_load(Path(args.batch_yaml).read_text()) or {}
            if isinstance(payload, dict):
                if payload.get("input_pdb") and not args.input_pdb:
                    args.input_pdb = payload.get("input_pdb")
                if payload.get("input_pdb_dir") and not args.input_pdb_dir:
                    args.input_pdb_dir = payload.get("input_pdb_dir")
        except Exception:
            pass

    output_dir = Path(args.output_dir).resolve()

    input_pdb_dir: Path | None = None
    if args.input_pdb:
        input_pdb = Path(args.input_pdb).resolve()
        if not input_pdb.exists():
            raise FileNotFoundError(f"input_pdb not found: {input_pdb}")
        single_dir = output_dir / "input_pdbs"
        single_dir.mkdir(parents=True, exist_ok=True)
        dst = single_dir / input_pdb.name
        if not dst.exists():
            try:
                os.link(input_pdb, dst)
            except Exception:
                shutil.copy2(input_pdb, dst)
        input_pdb_dir = single_dir
        args.num_jobs = 1
    else:
        if not args.input_pdb_dir:
            raise RuntimeError("--input_pdb_dir is required when --input_pdb is not provided")
        input_pdb_dir = Path(args.input_pdb_dir).resolve()

    input_pdb_dir = _resolve_input_dir(input_pdb_dir)
    af3_repo = Path(args.af3score_repo).resolve()
    model_dir = Path(args.model_dir).resolve()

    if not input_pdb_dir.exists():
        raise FileNotFoundError(f"input_pdb_dir not found: {input_pdb_dir}")
    if not af3_repo.exists():
        raise FileNotFoundError(f"af3score_repo not found: {af3_repo}")
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    af3_input_batch = output_dir / "af3_input_batch"
    output_dir_cif = output_dir / "single_chain_cif"
    output_dir_json = output_dir / "json"
    output_dir_jax = af3_input_batch / "jax"
    output_dir_af3score = output_dir / "af3score_outputs"
    metrics_csv = output_dir / "metrics.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    af3_input_batch.mkdir(parents=True, exist_ok=True)
    output_dir_cif.mkdir(parents=True, exist_ok=True)
    output_dir_json.mkdir(parents=True, exist_ok=True)
    output_dir_jax.mkdir(parents=True, exist_ok=True)
    output_dir_af3score.mkdir(parents=True, exist_ok=True)
    quiet = _quiet_enabled()
    subproc_log_dir = output_dir / "af3score_subprocess_logs" if quiet else None

    python_cmd = _resolve_python_cmd()
    env = os.environ.copy()
    # Force quiet featurisation logging in AF3Score via sitecustomize.
    patch_dir = (Path(__file__).resolve().parent / "af3score_sitecustomize")
    if patch_dir.exists():
        env["PYTHONPATH"] = str(patch_dir) + os.pathsep + env.get("PYTHONPATH", "")
    # Prefer AF3Score env binaries (ptxas, nvcc) over ppiflow env.
    if len(python_cmd) == 1:
        try:
            af3_bin = str(Path(python_cmd[0]).resolve().parent)
            env["PATH"] = af3_bin + os.pathsep + env.get("PATH", "")
        except Exception:
            pass
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
    # Ensure JAX uses a CUDA toolkit that matches its PTX requirements.
    cuda_home = (
        os.environ.get("AF3SCORE_CUDA_HOME")
        or os.environ.get("CUDA_HOME")
        or os.environ.get("CUDA_PATH")
    )
    if not cuda_home and len(python_cmd) == 1:
        try:
            cuda_home = str(Path(python_cmd[0]).resolve().parents[1])
        except Exception:
            cuda_home = None
    if cuda_home:
        current_flags = env.get("XLA_FLAGS", "")
        flag = f"--xla_gpu_cuda_data_dir={cuda_home}"
        if flag not in current_flags:
            env["XLA_FLAGS"] = (current_flags + " " + flag).strip()

    prep_json = af3_repo / "01_prepare_get_json.py"
    prep_jax = af3_repo / "02_prepare_pdb2jax.py"
    run_af3 = af3_repo / "run_af3score.py"
    get_metrics = af3_repo / "04_get_metrics.py"

    if not prep_json.exists() or not prep_jax.exists() or not run_af3.exists() or not get_metrics.exists():
        raise FileNotFoundError("AF3Score repo missing required scripts.")

    save_csv = output_dir / "single_seq.csv"
    start = time.time()
    status = "OK"
    try:
        _write_progress(output_dir, 0, 1, phase="prep_json", status="running")
        _run_cmd(
            python_cmd
            + [
                str(prep_json),
                "--input_dir",
                str(input_pdb_dir),
                "--output_dir_cif",
                str(output_dir_cif),
                "--save_csv",
                str(save_csv),
                "--output_dir_json",
                str(output_dir_json),
                "--batch_dir",
                str(af3_input_batch),
                "--num_jobs",
                str(args.num_jobs),
            ],
            env=env,
            cwd=str(output_dir),
            log_path=(subproc_log_dir / "prep_json.log") if subproc_log_dir else None,
        )
    except Exception:
        status = "FAILED"
        _write_progress(output_dir, 0, 1, phase="prep_json", status="failed")
        raise
    finally:
        _progress(1, 1, "prep_json", "all", status, time.time() - start)
        if status == "OK":
            _write_progress(output_dir, 1, 1, phase="prep_json", status="completed")

    model_seeds = _parse_seed_list(args.model_seeds)
    if args.no_templates or model_seeds:
        for json_path in sorted(output_dir_json.glob("*.json")):
            _strip_templates_and_set_seeds(
                json_path,
                no_templates=args.no_templates,
                model_seeds=model_seeds,
            )

    pdb_batches = sorted((af3_input_batch / "pdb").glob("*"))
    total_pdb = len(pdb_batches)
    for idx, subfolder in enumerate(pdb_batches, start=1):
        if not subfolder.is_dir():
            continue
        folder_name = subfolder.name
        out_h5 = output_dir_jax / folder_name
        out_h5.mkdir(parents=True, exist_ok=True)
        start = time.time()
        status = "OK"
        try:
            if idx == 1:
                _write_progress(output_dir, 0, total_pdb, phase="prep_jax", status="running")
            _run_cmd(
                python_cmd
                + [
                    str(prep_jax),
                    "--pdb_folder",
                    str(subfolder),
                    "--output_folder",
                    str(out_h5),
                    "--num_workers",
                    str(args.num_workers),
                ],
                env=env,
                cwd=str(output_dir),
                log_path=(subproc_log_dir / f"prep_jax_{folder_name}.log") if subproc_log_dir else None,
            )
        except Exception:
            status = "FAILED"
            _write_progress(output_dir, idx - 1, total_pdb, phase="prep_jax", item=folder_name, status="failed")
            raise
        finally:
            _progress(idx, total_pdb, "prep_jax", folder_name, status, time.time() - start)
            if status == "OK":
                phase_status = "completed" if idx == total_pdb else "running"
                _write_progress(output_dir, idx, total_pdb, phase="prep_jax", item=folder_name, status=phase_status)

    json_batches = sorted((af3_input_batch / "json").glob("*"))
    total_json = len(json_batches)
    for idx, subfolder in enumerate(json_batches, start=1):
        if not subfolder.is_dir():
            continue
        folder_name = subfolder.name
        batch_h5 = output_dir_jax / folder_name
        buckets = _bucket_from_name(folder_name, default=args.bucket_default)
        cmd = python_cmd + [
            str(run_af3),
            "--model_dir",
            str(model_dir),
            "--batch_json_dir",
            str(subfolder),
            "--batch_h5_dir",
            str(batch_h5),
            "--output_dir",
            str(output_dir_af3score),
            "--run_data_pipeline",
            "False",
            "--run_inference",
            "true",
            "--init_guess",
            "true",
            "--num_samples",
            str(args.num_samples),
            "--buckets",
            str(buckets),
            "--write_cif_model",
            "true" if str(args.write_cif_model).lower() in {"1", "true", "yes"} else "false",
            "--write_summary_confidences",
            "true",
            "--write_full_confidences",
            "true",
            "--write_best_model_root",
            "true" if args.write_best_model_root else "false",
            "--write_ranking_scores_csv",
            "true" if args.write_ranking_scores_csv else "false",
            "--write_terms_of_use_file",
            "false",
            "--write_fold_input_json_file",
            "false",
        ]
        if args.db_dir:
            cmd.extend(["--db_dir", str(Path(args.db_dir).resolve())])
        elif os.environ.get("AF3_DB_DIR"):
            cmd.extend(["--db_dir", os.environ["AF3_DB_DIR"]])
        start = time.time()
        status = "OK"
        try:
            if idx == 1:
                _write_progress(output_dir, 0, total_json, phase="run_af3", status="running")
            _run_cmd(
                cmd,
                env=env,
                cwd=str(output_dir),
                log_path=(subproc_log_dir / f"run_af3_{folder_name}.log") if subproc_log_dir else None,
            )
        except Exception:
            status = "FAILED"
            _write_progress(output_dir, idx - 1, total_json, phase="run_af3", item=folder_name, status="failed")
            raise
        finally:
            _progress(idx, total_json, "run_af3", folder_name, status, time.time() - start)
            if status == "OK":
                phase_status = "completed" if idx == total_json else "running"
                _write_progress(output_dir, idx, total_json, phase="run_af3", item=folder_name, status=phase_status)

    metrics_input_dir = _prepare_metrics_pdb_dir(input_pdb_dir, output_dir_af3score, output_dir)
    _ensure_seed10_alias(output_dir_af3score)
    start = time.time()
    status = "OK"
    try:
        _write_progress(output_dir, 0, 1, phase="metrics", status="running")
        _run_cmd(
            python_cmd
            + [
                str(get_metrics),
                "--input_pdb_dir",
                str(metrics_input_dir),
                "--af3score_output_dir",
                str(output_dir_af3score),
                "--save_metric_csv",
                str(metrics_csv),
            ],
            env=env,
            cwd=str(output_dir),
            log_path=(subproc_log_dir / "metrics.log") if subproc_log_dir else None,
        )
    except Exception:
        status = "FAILED"
        _write_progress(output_dir, 0, 1, phase="metrics", status="failed")
        raise
    finally:
        _progress(1, 1, "metrics", "all", status, time.time() - start)
        if status == "OK":
            _write_progress(output_dir, 1, 1, phase="metrics", status="completed")

    offset_map = None
    if args.target_offsets_json:
        offsets_path = Path(args.target_offsets_json).resolve()
        if offsets_path.exists():
            offset_map = _load_chain_offset_map(offsets_path)

    if args.export_pdb_dir:
        export_dir = Path(args.export_pdb_dir).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        for job_dir in sorted(output_dir_af3score.iterdir()):
            if not job_dir.is_dir():
                continue
            job_name = job_dir.name
            cif_path = job_dir / f"{job_name}_model.cif"
            if not cif_path.exists():
                seed_cifs = sorted(job_dir.glob("seed-*/*/model.cif"))
                if not seed_cifs:
                    seed_cifs = sorted(job_dir.glob("seed-*/model.cif"))
                if seed_cifs:
                    cif_path = seed_cifs[0]
                else:
                    continue
            pdb_path = export_dir / f"{job_name}.pdb"
            if pdb_path.exists():
                continue
            if _convert_cif_to_pdb(cif_path, pdb_path) and offset_map:
                _renumber_chain_with_offsets(pdb_path, args.target_chain, offset_map)


if __name__ == "__main__":
    main()
