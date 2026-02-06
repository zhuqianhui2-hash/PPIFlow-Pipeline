from __future__ import annotations

import os
import sys
from pathlib import Path

from .base import Step, StepContext
from ..work_queue import WorkItem
from ..io import collect_pdbs
from ..logging_utils import run_command
from ..manifests import extract_design_id, structure_id_from_name, write_csv


class DockQStep(Step):
    name = "dockq"
    stage = "score"
    supports_indices = False
    supports_work_queue = True
    work_queue_mode = "items"

    def expected_total(self, ctx: StepContext) -> int:
        return 1

    def scan_done(self, ctx: StepContext) -> set[int]:
        manifest = self.cfg.get("manifest")
        if manifest:
            try:
                import csv
                from pathlib import Path

                p = Path(manifest)
                if not p.is_absolute():
                    p = ctx.out_dir / p
                if p.exists():
                    with p.open("r", newline="") as handle:
                        rows = list(csv.reader(handle))
                    if len(rows) > 1:
                        return {0}
            except Exception:
                pass
        out_dir = self.output_dir(ctx)
        if list(out_dir.rglob("*_dockq_score")):
            return {0}
        return set()

    def run_full(self, ctx: StepContext) -> None:
        cmd = self.cfg.get("command")
        if not cmd:
            return
        if isinstance(cmd, str):
            cmd = [cmd]
        run_command(
            cmd,
            env=os.environ.copy(),
            log_file=self.cfg.get("_log_file"),
            verbose=bool(self.cfg.get("_verbose")),
        )

    def _dockq_config(self) -> dict:
        cfg = dict(self.cfg.get("dockq") or {})
        if cfg:
            return cfg
        cmd = self.cfg.get("command") or []
        if isinstance(cmd, str):
            cmd = [cmd]
        out: dict[str, object] = {}
        for idx, token in enumerate(cmd):
            if token in {"--dockq_bin", "--input_pdb_dir", "--reference_pdb_dir", "--output_dir", "--allowed_mismatches"}:
                if idx + 1 < len(cmd):
                    out[token.lstrip("-")] = cmd[idx + 1]
            elif token == "--skip_existing":
                out["skip_existing"] = True
        return out

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        cfg = self._dockq_config()
        input_dir = cfg.get("input_pdb_dir")
        if not input_dir:
            return []
        input_path = Path(str(input_dir))
        if not input_path.is_absolute():
            input_path = ctx.out_dir / input_path
        models = collect_pdbs(input_path)
        items: list[WorkItem] = []
        for model in models:
            rel = model.relative_to(input_path).as_posix()
            item_id = rel.replace("/", "__")
            items.append(
                WorkItem(
                    id=item_id,
                    payload={
                        "model_path": str(model),
                        "model_name": model.stem,
                    },
                )
            )
        return items

    def item_done(self, ctx: StepContext, item: WorkItem) -> bool:
        cfg = self._dockq_config()
        output_dir = cfg.get("output_dir")
        if not output_dir:
            return False
        out_path = Path(str(output_dir))
        if not out_path.is_absolute():
            out_path = ctx.out_dir / out_path
        model_name = str((item.payload or {}).get("model_name") or "")
        if not model_name:
            raw_id = str(getattr(item, "id", "") or "")
            base = raw_id.split("__")[-1]
            model_name = Path(base).stem
        return (out_path / f"{model_name}_dockq_score").exists()

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        cfg = self._dockq_config()
        dockq_bin = cfg.get("dockq_bin")
        input_pdb = (item.payload or {}).get("model_path")
        reference_dir = cfg.get("reference_pdb_dir")
        output_dir = cfg.get("output_dir")
        allowed = cfg.get("allowed_mismatches") or 10
        skip_existing = bool(cfg.get("skip_existing"))
        if not dockq_bin or not input_pdb or not reference_dir or not output_dir:
            raise RuntimeError("dockq config missing required fields")
        script = Path(__file__).resolve().parents[3] / "scripts" / "run_dockq.py"
        ref_path = Path(str(reference_dir))
        if not ref_path.is_absolute():
            ref_path = ctx.out_dir / ref_path
        out_path = Path(str(output_dir))
        if not out_path.is_absolute():
            out_path = ctx.out_dir / out_path
        cmd = [
            sys.executable,
            str(script),
            "--dockq_bin",
            str(dockq_bin),
            "--input_pdb",
            str(input_pdb),
            "--reference_pdb_dir",
            str(ref_path),
            "--output_dir",
            str(out_path),
            "--allowed_mismatches",
            str(int(allowed)),
        ]
        if skip_existing:
            cmd.append("--skip_existing")
        run_command(
            cmd,
            env=os.environ.copy(),
            log_file=self.cfg.get("_log_file"),
            verbose=bool(self.cfg.get("_verbose")),
        )

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        rows = []
        for fp in out_dir.rglob("*_dockq_score"):
            try:
                text = fp.read_text().strip().split()
                score = float(text[-1]) if text else None
            except Exception:
                score = None
            name = fp.stem.replace("_dockq_score", "")
            rows.append({
                "design_id": extract_design_id(name),
                "structure_id": structure_id_from_name(name),
                "dockq": score,
                "path": str(fp),
            })
        if not rows:
            return
        # apply filter if configured
        dockq_min = (ctx.input_data.get("filters") or {}).get("dockq", {}).get("min")
        if dockq_min is not None:
            dockq_min = float(dockq_min)
            for r in rows:
                r["passed_filter"] = r.get("dockq") is not None and float(r.get("dockq") or 0) >= dockq_min
        write_csv(self.manifest_path(ctx), rows, ["design_id", "structure_id", "dockq", "path", "passed_filter"])
