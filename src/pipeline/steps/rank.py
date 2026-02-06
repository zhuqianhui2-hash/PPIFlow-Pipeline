from __future__ import annotations

import csv
import filecmp
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .base import Step, StepContext, StepError
from ..io import ensure_dir, read_json, write_json
from ..work_queue import WorkItem


class RankStep(Step):
    name = "rank"
    stage = "rank"
    supports_indices = False
    supports_work_queue = True
    work_queue_mode = "leader"

    def expected_total(self, ctx: StepContext) -> int:
        sampling = ctx.input_data.get("sampling") or {}
        return int(sampling.get("samples_per_target", 0) or 0)

    def _results_dir(self, out_dir: Path, version: int) -> Path:
        if version <= 1:
            return out_dir / "results"
        return out_dir / f"results_v{version}"

    def _allocate_results_dir(self, out_dir: Path) -> Path:
        for v in range(1, 10000):
            p = self._results_dir(out_dir, v)
            if not p.exists():
                return p
            manifest = p / "manifest.json"
            if not manifest.exists():
                return p
            try:
                data = read_json(manifest)
            except Exception:
                return p
            if str((data or {}).get("status", "")).lower() != "completed":
                return p
        raise StepError("Too many results_v* directories; please clean up old results.")

    def _latest_completed_results_dir(self, out_dir: Path) -> Path | None:
        candidates: list[tuple[int, Path]] = []
        for path in sorted(out_dir.glob("results*")):
            if not path.is_dir():
                continue
            if path.name == "results":
                version = 1
            elif path.name.startswith("results_v"):
                try:
                    version = int(path.name.replace("results_v", ""))
                except Exception:
                    continue
            else:
                continue
            manifest = path / "manifest.json"
            summary = path / "summary.csv"
            if not manifest.exists() or not summary.exists():
                continue
            try:
                data = read_json(manifest) or {}
            except Exception:
                continue
            if str(data.get("status") or "").lower() != "completed":
                continue
            candidates.append((version, path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    def _output_meta_dir(self, ctx: StepContext) -> Path | None:
        results_dir = getattr(self, "_results_dir", None)
        if isinstance(results_dir, Path):
            return results_dir
        return super()._output_meta_dir(ctx)

    def outputs_complete(self, ctx: StepContext) -> bool:
        results_dir = self._latest_completed_results_dir(ctx.out_dir)
        if results_dir is None:
            return False
        self._results_dir = results_dir
        meta = self._load_output_meta(ctx, meta_dir=results_dir)
        if meta is None:
            if not self._allow_legacy_outputs(ctx):
                return False
            self._warn("missing output metadata; accepting legacy outputs due to explicit reuse")
        else:
            if not self._validate_output_meta(ctx, meta, meta_dir=results_dir):
                return False
        return True

    def _find_metrics_csv(self, run_dir: Path) -> Path | None:
        for rel in [
            "af3_refold/metrics_ppiflow.csv",
            "af3_refold/metrics.csv",
            "af3score_round2/metrics_ppiflow.csv",
            "af3score_round1/metrics_ppiflow.csv",
            "af3score_round2/metrics.csv",
            "af3score_round1/metrics.csv",
        ]:
            p = run_dir / rel
            if p.exists():
                return p
        return None

    def _pick_ptm_column(self, columns: list[str]) -> str | None:
        for c in ["ptm", "pTM", "ptm_A", "ptm_B", "AF3Score_chain_ptm"]:
            if c in columns:
                return c
        return None

    def _pick_score_column(
        self,
        columns: list[str],
        *,
        df=None,
        protocol: str | None = None,
    ) -> str | None:
        if protocol == "antibody":
            candidates = [
                "iptm_binder_target",
                "AF3Score_interchain_iptm",
                "AF3Score_chain_iptm",
                "iptm",
                "ipTM",
                "ptm",
                "pTM",
            ]
        else:
            candidates = [
                "AF3Score_interchain_iptm",
                "AF3Score_chain_iptm",
                "iptm",
                "ipTM",
                "ptm",
                "pTM",
                "iptm_binder_target",
            ]
        for c in candidates:
            if c in columns:
                if df is not None:
                    try:
                        if df[c].notna().any():
                            return c
                    except Exception:
                        return c
                else:
                    return c
        return None

    def _load_interface_scores(self, run_dir: Path) -> dict[str, float]:
        # Interface scores must come from post-round2 analysis only.
        candidates = list(run_dir.glob("rosetta_interface2/*.csv"))
        mapping: dict[str, float] = {}
        for path in candidates:
            try:
                import pandas as pd

                df = pd.read_csv(path)
            except Exception:
                continue
            score_col = None
            for col in ["interface_score", "I_sc", "interface_delta", "total_score", "score"]:
                if col in df.columns:
                    score_col = col
                    break
            name_col = None
            for col in ["description", "name", "pdb_name", "model"]:
                if col in df.columns:
                    name_col = col
                    break
            if not score_col or not name_col:
                continue
            for _, row in df.iterrows():
                name = str(row.get(name_col))
                try:
                    mapping[name] = float(row.get(score_col))
                except Exception:
                    continue
        return mapping

    def _lookup_interface_score(self, name: str, mapping: dict[str, float]) -> float | None:
        if not name:
            return None
        candidates = [name]
        candidates.append(name.lower())
        if re.search(r"_\d+$", name):
            candidates.append(re.sub(r"_\d+$", "", name))
        if "__sample" in name:
            # Map refold names to round1-style names (e.g. foo__sample3_4 -> foo_4).
            candidates.append(re.sub(r"__sample\d+_", "_", name))
            candidates.append(re.sub(r"__sample\d+", "", name))
        if "__" in name:
            candidates.append(name.replace("__", "_"))
        for cand in candidates:
            if cand in mapping:
                return mapping[cand]
        return None

    def _augment_binder_interface_scores(self, run_dir: Path, mapping: dict[str, float]) -> dict[str, float]:
        partial_root = run_dir / "partial_flow"
        seqs_pdb_dir = run_dir / "seqs_round2" / "pdbs"
        if not partial_root.exists() or not seqs_pdb_dir.exists():
            return mapping

        def _raw_path_for_partial(base_dir: Path) -> str | None:
            input_dir = base_dir / "input"
            if not input_dir.exists():
                return None
            for csv_path in sorted(input_dir.glob("*_input.csv")):
                try:
                    with open(csv_path, newline="") as handle:
                        reader = csv.DictReader(handle)
                        row = next(reader, None)
                    if row and row.get("raw_path"):
                        return row["raw_path"]
                except Exception:
                    continue
            return None

        augmented = dict(mapping)
        for seq_pdb in sorted(seqs_pdb_dir.glob("*.pdb")):
            name = seq_pdb.stem
            if "__" not in name:
                continue
            prefix, sample = name.split("__", 1)
            sample_file = f"{sample}.pdb"
            matched_score = None
            for base_dir in sorted(partial_root.iterdir()):
                if not base_dir.is_dir():
                    continue
                cand = base_dir / prefix / sample_file
                if not cand.exists():
                    continue
                try:
                    same = filecmp.cmp(seq_pdb, cand, shallow=False)
                except Exception:
                    same = True
                if not same:
                    continue
                raw_path = _raw_path_for_partial(base_dir)
                if not raw_path:
                    continue
                score = augmented.get(Path(raw_path).stem)
                if score is not None:
                    matched_score = score
                    break
            if matched_score is not None:
                augmented[name] = matched_score
                augmented[name.lower()] = matched_score
        return augmented

    def _load_dockq_scores(self, run_dir: Path) -> dict[str, float]:
        dockq_dir = run_dir / "dockq"
        mapping: dict[str, float] = {}
        if not dockq_dir.exists():
            return mapping
        for fp in dockq_dir.glob("*_dockq_score"):
            name = fp.stem
            if name.endswith("_dockq_score"):
                name = name[: -len("_dockq_score")]
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
            if score is not None:
                mapping[name] = score
        return mapping

    def _collect_structures(self, run_dir: Path, name: str) -> tuple[dict[int, Path], dict[str, Path]]:
        # Priority: refold -> relax -> partial_flow -> backbones
        mapping_by_id: dict[int, Path] = {}
        mapping_by_name: dict[str, Path] = {}
        for rel, pattern in [
            ("af3_refold/pdbs", "*.pdb"),
            ("relax", "*.pdb"),
            ("partial_flow", "**/*.pdb"),
            ("backbones", f"{name}_*.pdb"),
        ]:
            root = run_dir / rel
            if not root.exists():
                continue
            for fp in root.glob(pattern):
                stem = fp.stem
                mapping_by_name.setdefault(stem, fp)
                if stem.startswith(f"{name}_"):
                    suffix = stem.split("_", 1)[1]
                    if suffix.isdigit():
                        mapping_by_id.setdefault(int(suffix), fp)
                elif stem.startswith("sample"):
                    # partial flow naming sample<id>_*
                    rest = stem.replace("sample", "", 1)
                    parts = rest.split("_", 1)
                    if parts and parts[0].isdigit():
                        mapping_by_id.setdefault(int(parts[0]), fp)
        return mapping_by_id, mapping_by_name

    def _compute_rows(self, ctx: StepContext) -> tuple[list[dict[str, Any]], bool]:
        out_dir = ctx.out_dir
        run_dir = out_dir / "output"
        name = ctx.input_data.get("name", "design")
        metrics_path = self._find_metrics_csv(run_dir)

        rows: list[dict[str, Any]] = []
        interface_scores = self._load_interface_scores(run_dir)
        protocol = str(ctx.input_data.get("protocol") or "")
        if protocol == "binder":
            interface_scores = self._augment_binder_interface_scores(run_dir, interface_scores)
        dockq_scores = self._load_dockq_scores(run_dir)
        using_refold = bool(metrics_path and "af3_refold" in str(metrics_path))

        if metrics_path and metrics_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(metrics_path)
                score_col = self._pick_score_column(list(df.columns), df=df, protocol=protocol)
                ptm_col = self._pick_ptm_column(list(df.columns))
                if score_col:
                    df = df.sort_values(score_col, ascending=False)
                else:
                    df = df.sort_values(df.columns[0])
                for _, row in df.iterrows():
                    design_name = None
                    if "design_id" in row and str(row.get("design_id") or "").strip():
                        design_name = row.get("design_id")
                    elif "description" in row and str(row.get("description") or "").strip():
                        design_name = row.get("description")
                    else:
                        design_name = row.get("name", None)
                    iptm_val = row.get(score_col) if score_col else None
                    ptm_val = row.get(ptm_col) if ptm_col else None
                    dockq_val = dockq_scores.get(str(design_name))
                    rows.append({
                        "design_id": design_name,
                        "score": iptm_val,
                        "iptm": iptm_val,
                        "ptm": ptm_val,
                        "dockq": dockq_val,
                        "interface_score": self._lookup_interface_score(str(design_name), interface_scores),
                    })
            except Exception:
                rows = []

        if not rows:
            # Fallback to file listing
            mapping_by_id, _ = self._collect_structures(run_dir, name)
            for design_id in sorted(mapping_by_id.keys()):
                rows.append({"design_id": design_id, "score": None, "interface_score": None})

        mapping_by_id, mapping_by_name = self._collect_structures(run_dir, name)
        for r in rows:
            did = r.get("design_id")
            if did is None or (isinstance(did, float) and str(did) == "nan"):
                continue
            try:
                did_int = int(did)
            except Exception:
                did_int = None
            if did_int is not None and did_int in mapping_by_id:
                r["source_path"] = str(mapping_by_id[did_int])
            else:
                key = str(did)
                if key in mapping_by_name:
                    r["source_path"] = str(mapping_by_name[key])

        filters = ctx.input_data.get("filters") or {}
        if using_refold:
            refold = filters.get("af3_refold") or {}
            iptm_min = refold.get("iptm_min")
            ptm_min = refold.get("ptm_min")
            dockq_min = refold.get("dockq_min")
        else:
            r2 = (filters.get("af3score") or {}).get("round2") or {}
            iptm_min = r2.get("iptm_min")
            ptm_min = r2.get("ptm_min")
            dockq_min = (filters.get("dockq") or {}).get("min")

        for r in rows:
            passed = True
            try:
                if iptm_min is not None:
                    passed = passed and r.get("iptm") is not None and float(r.get("iptm")) >= float(iptm_min)
                if ptm_min is not None:
                    passed = passed and r.get("ptm") is not None and float(r.get("ptm")) >= float(ptm_min)
                if dockq_min is not None:
                    passed = passed and r.get("dockq") is not None and float(r.get("dockq")) >= float(dockq_min)
            except Exception:
                passed = False
            r["passed_filter"] = passed

        for r in rows:
            try:
                iptm = float(r.get("iptm")) if r.get("iptm") is not None else None
                interface_score = float(r.get("interface_score")) if r.get("interface_score") is not None else None
            except Exception:
                iptm = None
                interface_score = None
            if iptm is not None and interface_score is not None:
                r["composite_score"] = iptm * 100.0 - interface_score
            else:
                r["composite_score"] = None

        return rows, using_refold

    def run_full(self, ctx: StepContext) -> None:
        out_dir = ctx.out_dir
        results_dir = self._allocate_results_dir(out_dir)
        self._results_dir = results_dir
        ensure_dir(results_dir)
        structures_dir = results_dir / "structures"
        top_dir = structures_dir / "top"
        ensure_dir(top_dir)
        rows: list[dict[str, Any]] = []
        features_dir = self.cfg.get("features_dir")
        if features_dir:
            p = Path(str(features_dir))
            if not p.is_absolute():
                p = ctx.out_dir / p
            if p.exists():
                for fp in sorted(p.glob("*.json")):
                    try:
                        data = read_json(fp)
                        if isinstance(data, dict):
                            rows.append(data)
                    except Exception:
                        continue
        if not rows:
            rows, _ = self._compute_rows(ctx)

        ranking_rows = [r for r in rows if r.get("passed_filter")]
        if not ranking_rows:
            ranking_rows = rows

        ranking_rows.sort(
            key=lambda x: (x.get("composite_score") is not None, x.get("composite_score", 0)),
            reverse=True,
        )

        summary_path = results_dir / "summary.csv"
        tmp_path = summary_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "design_id",
                    "score",
                    "iptm",
                    "ptm",
                    "dockq",
                    "interface_score",
                    "composite_score",
                    "passed_filter",
                    "source_path",
                ],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "design_id": r.get("design_id"),
                    "score": r.get("score"),
                    "iptm": r.get("iptm"),
                    "ptm": r.get("ptm"),
                    "dockq": r.get("dockq"),
                    "interface_score": r.get("interface_score"),
                    "composite_score": r.get("composite_score"),
                    "passed_filter": r.get("passed_filter"),
                    "source_path": r.get("source_path"),
                })
        tmp_path.replace(summary_path)

        # Copy top structures
        ranking_cfg = self.cfg.get("ranking") or {}
        top_k = int(ranking_cfg.get("top_k", 30) or 30)
        count = 0
        for r in ranking_rows:
            if count >= top_k:
                break
            src = r.get("source_path")
            if not src:
                continue
            src_path = Path(src)
            if not src_path.exists():
                continue
            dst = top_dir / src_path.name
            if not dst.exists():
                try:
                    os.link(src_path, dst)
                except Exception:
                    dst.write_bytes(src_path.read_bytes())
            count += 1

        write_json(
            results_dir / "manifest.json",
            {
                "status": "completed",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "step": "rank",
                "run_id": ctx.run_id,
            },
            indent=2,
        )


class RankFeaturesStep(RankStep):
    name = "rank_features"
    stage = "rank"
    supports_indices = False
    supports_work_queue = True
    work_queue_mode = "items"

    def expected_total(self, ctx: StepContext) -> int:
        return 1

    def build_items(self, ctx: StepContext) -> list[WorkItem]:
        rows, _ = self._compute_rows(ctx)
        items: list[WorkItem] = []
        seen: set[str] = set()
        for idx, row in enumerate(rows):
            raw_id = row.get("design_id")
            base_id = str(raw_id) if raw_id is not None and str(raw_id) != "nan" else f"row_{idx}"
            item_id = base_id
            if item_id in seen:
                item_id = f"{base_id}_{idx}"
            seen.add(item_id)
            payload = dict(row)
            payload["design_id"] = raw_id
            items.append(WorkItem(id=item_id, payload=payload))
        return items

    def item_done(self, ctx: StepContext, item: WorkItem) -> bool:
        out_dir = self.output_dir(ctx)
        return (out_dir / f"{item.id}.json").exists()

    def run_item(self, ctx: StepContext, item: WorkItem) -> None:
        out_dir = self.output_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _coerce(val: Any) -> Any:
            if val is None:
                return None
            if isinstance(val, (str, int, float, bool)):
                return val
            try:
                # numpy scalars
                return val.item()
            except Exception:
                pass
            if isinstance(val, Path):
                return str(val)
            return str(val)

        payload = {k: _coerce(v) for k, v in (item.payload or {}).items()}
        write_json(out_dir / f"{item.id}.json", payload, indent=2)
