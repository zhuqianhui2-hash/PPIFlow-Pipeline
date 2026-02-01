#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _dockq_cmd(dockq_bin: str) -> list[str]:
    if dockq_bin.endswith(".py"):
        return [sys.executable, dockq_bin]
    return [dockq_bin]


def _write_progress(output_dir: Path, produced: int, expected: int, *, item: str | None = None, status: str = "running") -> None:
    payload = {
        "expected_total": max(int(expected), 0),
        "produced_total": max(int(produced), 0),
        "status": status,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DockQ over a directory of PDBs")
    parser.add_argument("--dockq_bin", required=True, help="Path to DockQ executable or DockQ.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_pdb_dir", help="Directory with model PDBs (e.g., relax outputs)")
    group.add_argument("--input_pdb", help="Single model PDB path")
    parser.add_argument("--reference_pdb_dir", required=True, help="Directory with reference PDBs")
    parser.add_argument("--output_dir", required=True, help="Directory to write *_dockq_score files")
    parser.add_argument("--allowed_mismatches", type=int, default=10, help="DockQ allowed mismatches")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output exists")

    args = parser.parse_args()
    input_dir = Path(args.input_pdb_dir) if args.input_pdb_dir else None
    ref_dir = Path(args.reference_pdb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_dir and not input_dir.exists():
        raise FileNotFoundError(f"input_pdb_dir not found: {input_dir}")

    dockq_cmd = _dockq_cmd(args.dockq_bin)

    failures: list[str] = []
    if args.input_pdb:
        models = [Path(args.input_pdb)]
    else:
        models = sorted(input_dir.glob("*.pdb")) if input_dir else []
    total = len(models)
    _write_progress(out_dir, 0, total, status="running")
    for idx, model_path in enumerate(models, start=1):
        if not model_path.exists():
            failures.append(f"missing_model\t{model_path}")
            _write_progress(out_dir, idx, total, item=str(model_path), status="running")
            continue
        ref_path = ref_dir / model_path.name
        if not ref_path.exists():
            failures.append(f"missing_reference\t{model_path.name}")
            _write_progress(out_dir, idx, total, item=model_path.name, status="running")
            continue
        out_path = out_dir / f"{model_path.stem}_dockq_score"
        if args.skip_existing and out_path.exists():
            _write_progress(out_dir, idx, total, item=model_path.name, status="running")
            continue
        cmd = (
            dockq_cmd
            + [
                "--allowed_mismatches",
                str(args.allowed_mismatches),
                str(model_path),
                str(ref_path),
                "--short",
            ]
        )
        try:
            with out_path.open("w") as handle:
                res = subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                failures.append(f"dockq_failed\t{model_path.name}\t{res.stderr.strip()}")
        except Exception as exc:
            failures.append(f"dockq_exception\t{model_path.name}\t{exc}")
        _write_progress(out_dir, idx, total, item=model_path.name, status="running")

    if failures:
        fail_path = out_dir / "failed_records.txt"
        fail_path.write_text("\n".join(failures) + "\n")
        _write_progress(out_dir, total, total, status="failed")
        return 1
    _write_progress(out_dir, total, total, status="completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
