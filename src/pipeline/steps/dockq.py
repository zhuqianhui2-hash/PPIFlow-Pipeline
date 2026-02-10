from __future__ import annotations

import os
import sys
from pathlib import Path

from .base import Step, StepContext
from ..metrics_ledger import MetricsLedger
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
        score_path = out_path / f"{model_name}_dockq_score"
        if not score_path.exists():
            return False
        ledger = MetricsLedger(ctx.out_dir, self.output_dir(ctx))
        try:
            return ledger.has_done(str(item.id))
        finally:
            ledger.close()

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
        # Record score in ledger (crash-safe, multi-worker safe).
        model_name = str((item.payload or {}).get("model_name") or "")
        if not model_name:
            raw_id = str(getattr(item, "id", "") or "")
            base = raw_id.split("__")[-1]
            model_name = Path(base).stem
        score_path = out_path / f"{model_name}_dockq_score"
        score = None
        try:
            for line in score_path.read_text().splitlines():
                line = line.strip()
                if not line.startswith("DockQ"):
                    continue
                parts = line.split()
                if len(parts) > 1:
                    score = float(parts[1])
                    break
        except Exception:
            score = None
        ledger = MetricsLedger(ctx.out_dir, self.output_dir(ctx))
        try:
            ledger.upsert(
                str(item.id),
                status="done",
                metrics={"dockq": score},
                outputs={"path": str(score_path), "model_path": str(input_pdb)},
                worker_id=MetricsLedger.default_worker_id(),
                attempt=int((item.payload or {}).get("_attempt") or 1) if isinstance(item.payload, dict) else 1,
                design_id=extract_design_id(model_name),
                structure_id=structure_id_from_name(model_name),
            )
        finally:
            ledger.close()

    def write_manifest(self, ctx: StepContext) -> None:
        out_dir = self.output_dir(ctx)
        # Map model_name (score filename stem) -> WorkQueue item id for consistent resume semantics.
        item_id_by_model_name: dict[str, str] = {}
        try:
            for it in self.build_items(ctx):
                try:
                    mname = str((it.payload or {}).get("model_name") or "")
                except Exception:
                    mname = ""
                if mname:
                    item_id_by_model_name[mname] = str(it.id)
        except Exception:
            item_id_by_model_name = {}

        rows = []
        # Prefer ledger (canonical). Fallback to scanning raw score files for legacy runs.
        try:
            ledger = MetricsLedger(ctx.out_dir, out_dir)
            try:
                for r in ledger.iter_rows(status="done"):
                    score = None
                    if isinstance(r.metrics, dict):
                        score = r.metrics.get("dockq")
                    path = None
                    if isinstance(r.outputs, dict):
                        path = r.outputs.get("path")
                    rows.append({
                        "design_id": r.design_id if r.design_id is not None else extract_design_id(r.item_id),
                        "structure_id": r.structure_id or structure_id_from_name(r.item_id),
                        "dockq": score,
                        "path": str(path) if path else None,
                    })
            finally:
                ledger.close()
        except Exception:
            rows = []

        if not rows:
            # Reconcile: if raw score files exist but the ledger is missing/incomplete, backfill.
            try:
                ledger = MetricsLedger(ctx.out_dir, out_dir)
                try:
                    for fp in out_dir.rglob("*_dockq_score"):
                        fname = fp.name
                        model_name = fname[: -len("_dockq_score")] if fname.endswith("_dockq_score") else fname
                        # IMPORTANT: The WorkQueue item.id is not the same as model_name (item.id is a relpath
                        # encoded with "__"). Only backfill when we can map score -> WorkItem.id, otherwise
                        # item_done() would never observe the row and resume semantics break.
                        item_id = item_id_by_model_name.get(model_name)
                        if not item_id:
                            continue
                        if ledger.has_done(item_id):
                            continue
                        score = None
                        try:
                            for line in fp.read_text().splitlines():
                                line = line.strip()
                                if not line.startswith("DockQ"):
                                    continue
                                parts = line.split()
                                if len(parts) > 1:
                                    score = float(parts[1])
                                    break
                        except Exception:
                            score = None
                        ledger.upsert(
                            item_id,
                            status="done",
                            metrics={"dockq": score},
                            outputs={"path": str(fp), "model_name": model_name},
                            worker_id="reconcile",
                            attempt=1,
                            design_id=extract_design_id(model_name),
                            structure_id=structure_id_from_name(model_name),
                        )
                finally:
                    ledger.close()
            except Exception:
                pass
            for fp in out_dir.rglob("*_dockq_score"):
                # NOTE: DockQ score files are created from model_path.stem where model_path is a PDB like:
                #   <id>.pdb__sampleX_-1_Y.pdb
                # so the score filename contains an internal ".pdb__..." segment. Using Path.stem would
                # truncate at that dot and break ID matching. Use the literal filename instead.
                fname = fp.name
                model_name = fname[: -len("_dockq_score")] if fname.endswith("_dockq_score") else fname
                score = None
                try:
                    for line in fp.read_text().splitlines():
                        line = line.strip()
                        if not line.startswith("DockQ"):
                            continue
                        parts = line.split()
                        if len(parts) > 1:
                            score = float(parts[1])
                            break
                except Exception:
                    score = None
                rows.append({
                    "design_id": extract_design_id(model_name),
                    "structure_id": structure_id_from_name(model_name),
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

    def outputs_complete(self, ctx: StepContext) -> bool:
        if not super().outputs_complete(ctx):
            return False
        out_dir = self._resolve_output_dir_path(ctx)
        if not (out_dir / "metrics.db").exists():
            if not self._allow_legacy_outputs(ctx):
                return False
        if not list(out_dir.rglob("*_dockq_score")):
            return False
        return True

    def _finalize_work_queue_outputs(self, ctx: StepContext, wq, *, items: list[object], allow_failures: bool) -> None:
        # Materialize manifest + seal WAL before writing metadata.
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
            try:
                ledger = MetricsLedger(ctx.out_dir, self.output_dir(ctx))
                try:
                    ledger.checkpoint_and_truncate_wal()
                finally:
                    ledger.close()
            except Exception:
                pass
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
