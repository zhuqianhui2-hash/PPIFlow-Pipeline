from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import PDB
from Bio.SeqUtils import seq1

from .base import Step, StepContext, StepError
from ..manifests import structure_id_from_name, write_csv

_ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def _format_res_id(res: PDB.Residue.Residue) -> str:
    _, num, insert = res.get_id()
    return f"{num}{insert.strip()}"


def _chain_residue_map(pdb_path: Path, chain_id: str) -> dict[int, str]:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]
    if chain_id not in model:
        raise StepError(f"Chain {chain_id} not found in {pdb_path}")
    chain = model[chain_id]
    mapping: dict[int, str] = {}
    for res in chain:
        if not PDB.is_aa(res, standard=True):
            continue
        try:
            aa = seq1(res.get_resname())
        except Exception:
            continue
        resseq = int(res.get_id()[1])
        if resseq not in mapping:
            mapping[resseq] = aa
    return mapping


def _parse_indices(value: str) -> list[int]:
    if not value:
        return []
    tokens = str(value).strip().split()
    indices: list[int] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            indices.append(int(token))
            continue
        except Exception:
            pass
        match = re.match(r"(\d+)", token)
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def _indices_to_string(indices: list[int], chain_id: str, output_format: str) -> str:
    if not indices:
        return ""
    if output_format == "pdb":
        return ",".join(f"{chain_id}{i}" for i in indices)
    return " ".join(str(i) for i in indices)


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


def _write_enriched_pdb(
    src: Path,
    dst: Path,
    binder_chain: str,
    key_res: dict[int, tuple[float, str]],
    target_chain: str | None = None,
    target_mapping: list[int] | None = None,
) -> None:
    lines = src.read_text().splitlines()
    out_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) > 26:
            chain_id = line[21]
            if chain_id == binder_chain:
                try:
                    resseq = int(line[22:26])
                except Exception:
                    resseq = None
                if resseq is not None and resseq in key_res:
                    aa = key_res[resseq][1]
                    new_res = _ONE_TO_THREE.get(aa.upper())
                    if new_res:
                        line = f"{line[:17]}{new_res:<3}{line[20:]}"
        out_lines.append(line)
    dst.write_text("\n".join(out_lines) + "\n")
    if target_chain and target_mapping:
        if not _renumber_chain_with_offsets(dst, target_chain, target_mapping):
            raise StepError(f"Failed to renumber target chain {target_chain} in {dst}")


def _collapse_ranges(indices: list[int], chain_id: str) -> str:
    if not indices:
        return ""
    indices = sorted(set(int(i) for i in indices))
    ranges: list[tuple[int, int]] = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append((start, prev))
        start = prev = i
    ranges.append((start, prev))
    parts = []
    for a, b in ranges:
        if a == b:
            parts.append(f"{chain_id}{a}")
        else:
            parts.append(f"{chain_id}{a}-{chain_id}{b}")
    return ",".join(parts)


def _extract_fw_cdr_positions(pdb_path: Path, chain_id: str) -> tuple[str, str]:
    try:
        from abnumber import Chain, ChainParseError
    except Exception as exc:
        raise StepError("abnumber is required for antibody/VHH CDR extraction") from exc

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]
    if chain_id not in model:
        raise StepError(f"Chain {chain_id} not found in {pdb_path}")
    chain = model[chain_id]

    residues = []
    seq_chars = []
    for res in chain:
        if PDB.is_aa(res, standard=True):
            try:
                aa = seq1(res.get_resname())
            except Exception:
                continue
            seq_chars.append(aa)
            residues.append(res)
    full_seq = "".join(seq_chars)
    if len(full_seq) < 50:
        raise StepError(f"Chain {chain_id} sequence too short for CDR extraction in {pdb_path}")

    try:
        ab_chain = Chain(full_seq, scheme="imgt", cdr_definition="imgt")
    except (ChainParseError, Exception) as exc:
        raise StepError(f"Failed to parse antibody chain for {pdb_path}") from exc

    v_seq_str = ab_chain.seq
    start_index = full_seq.find(v_seq_str)
    if start_index == -1:
        raise StepError(f"Failed to align antibody sequence for {pdb_path}")

    v_seq_positions = list(ab_chain.positions.keys())
    fw_indices: list[str] = []
    cdr_indices: list[str] = []

    regions_order = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    for region_name in regions_order:
        region_dict = getattr(ab_chain, f"{region_name.lower()}_dict", None)
        if not region_dict:
            continue
        is_cdr = "CDR" in region_name.upper()
        for pos_obj in region_dict.keys():
            try:
                idx_in_v = v_seq_positions.index(pos_obj)
                abs_idx = start_index + idx_in_v
                res = residues[abs_idx]
                res_str = _format_res_id(res)
                if is_cdr:
                    cdr_indices.append(res_str)
                else:
                    fw_indices.append(res_str)
            except (ValueError, IndexError):
                continue

    fw_index = " ".join(fw_indices)
    cdr_position = ",".join(f"{chain_id}{i}" for i in cdr_indices)
    return fw_index, cdr_position


class InterfaceEnrichStep(Step):
    name = "interface_enrich"
    stage = "rosetta"
    supports_indices = False
    supports_work_queue = True
    work_queue_mode = "leader"

    def expected_total(self, ctx: StepContext) -> int:
        return 1

    def scan_done(self, ctx: StepContext) -> set[int]:
        out_dir = self.output_dir(ctx)
        if (out_dir / "fixed_positions.csv").exists():
            return {0}
        return set()

    def _default_residue_energy_path(self, ctx: StepContext) -> Path:
        run_dir = ctx.out_dir / "output"
        return run_dir / "rosetta_interface" / "residue_energy.csv"

    def run_full(self, ctx: StepContext) -> None:
        residue_energy_path = self.cfg.get("residue_energy_csv")
        if residue_energy_path:
            p = Path(residue_energy_path)
            if not p.is_absolute():
                p = ctx.out_dir / p
        else:
            p = self._default_residue_energy_path(ctx)
        if not p.exists():
            raise StepError(f"residue_energy.csv not found at {p}")

        df = pd.read_csv(p)
        if "binder_energy" not in df.columns:
            raise StepError("residue_energy.csv missing binder_energy column")

        protocol = ctx.input_data.get("protocol")
        is_antibody = protocol in {"antibody", "vhh"}
        is_full_antibody = protocol == "antibody"
        binder_chain = str(ctx.input_data.get("binder_chain") or "A")
        if is_antibody:
            framework = ctx.input_data.get("framework") or {}
            binder_chain = str(framework.get("heavy_chain") or binder_chain)
            light_chain = framework.get("light_chain")
        else:
            light_chain = None

        cutoff = float((ctx.input_data.get("filters") or {}).get("rosetta", {}).get("interface_energy_min") or -5.0)

        key_res_by_sid: dict[str, dict[int, tuple[float, str]]] = {}
        meta_by_sid: dict[str, dict[str, Any]] = {}
        chain_cache: dict[tuple[str, str], dict[int, str]] = {}

        for _, row in df.iterrows():
            name = str(row.get("pdbname") or Path(str(row.get("pdbpath", ""))).stem)
            if not name:
                continue
            sid = structure_id_from_name(name)
            pdb_path_raw = row.get("pdbpath") or row.get("pdb_path")
            if not pdb_path_raw:
                continue
            pdb_path = Path(str(pdb_path_raw))
            if not pdb_path.is_absolute():
                pdb_path = (ctx.out_dir / pdb_path).resolve()
            if not pdb_path.exists():
                continue

            meta_by_sid.setdefault(sid, {
                "pdb_name": sid,
                "pdb_path": str(pdb_path),
                "binder_chain": binder_chain,
            })

            try:
                energy_dict = ast.literal_eval(row.get("binder_energy") or "{}")
            except Exception:
                energy_dict = {}

            cache_key = (str(pdb_path), binder_chain)
            if cache_key not in chain_cache:
                chain_cache[cache_key] = _chain_residue_map(pdb_path, binder_chain)
            residue_map = chain_cache[cache_key]

            for k, v in energy_dict.items():
                try:
                    pos = int(k)
                    energy = float(v)
                except Exception:
                    continue
                if energy >= cutoff:
                    continue
                aa = residue_map.get(pos)
                if not aa:
                    continue
                cur = key_res_by_sid.setdefault(sid, {})
                prev = cur.get(pos)
                if prev is None or energy < prev[0]:
                    cur[pos] = (energy, aa)

        if not meta_by_sid:
            raise StepError("No valid interface entries found in residue_energy.csv")

        output_dir = self.output_dir(ctx)
        merged_dir = output_dir / "merged_pdbs"
        merged_dir.mkdir(parents=True, exist_ok=True)

        target_offsets = (ctx.input_data.get("target") or {}).get("chain_offsets")
        target_mapping = _load_chain_offset_map(Path(str(target_offsets))) if target_offsets else None
        target_chain = "B" if target_mapping else None

        out_rows: list[dict[str, Any]] = []
        for sid, meta in meta_by_sid.items():
            key_res = key_res_by_sid.get(sid, {})
            positions = sorted(key_res.keys())
            key_res_index = " ".join(str(i) for i in positions)
            if not positions:
                print(f"[interface_enrich] WARN no key residues for {sid}; proceeding without fixed positions", flush=True)

            merged_path = merged_dir / f"{sid}.pdb"
            _write_enriched_pdb(
                Path(meta["pdb_path"]),
                merged_path,
                binder_chain,
                key_res,
                target_chain=target_chain,
                target_mapping=target_mapping,
            )

            fw_index = ""
            cdr_position = ""
            fixed_structure = _indices_to_string(positions, binder_chain, "pdb")
            fixed_sequence = fixed_structure
            fixed_sequence_pm = key_res_index

            if is_antibody:
                fw_index, cdr_position = _extract_fw_cdr_positions(merged_path, binder_chain)
                if is_full_antibody and light_chain:
                    # Multi-chain fixed sequence for full antibodies (heavy + light).
                    fixed_sequence_parts = []
                    if positions:
                        fixed_sequence_parts.append(
                            ",".join(f"{binder_chain}{i}" for i in positions)
                        )
                    if fw_index:
                        fixed_sequence_parts.append(
                            ",".join(f"{binder_chain}{tok}" for tok in fw_index.split())
                        )
                    cdr_parts = [cdr_position] if cdr_position else []

                    fw_index_l, cdr_position_l = _extract_fw_cdr_positions(merged_path, light_chain)
                    if fw_index_l:
                        fixed_sequence_parts.append(
                            ",".join(f"{light_chain}{tok}" for tok in fw_index_l.split())
                        )
                    if cdr_position_l:
                        cdr_parts.append(cdr_position_l)

                    fixed_sequence = ",".join([p for p in fixed_sequence_parts if p])
                    cdr_position = ",".join([p for p in cdr_parts if p])
                    fixed_sequence_pm = ""
                else:
                    combined = " ".join(x for x in [key_res_index, fw_index] if x).strip()
                    combined_indices = _parse_indices(combined)
                    fixed_sequence_pm = _indices_to_string(combined_indices, binder_chain, "space")
                    fixed_sequence = _indices_to_string(combined_indices, binder_chain, "pdb")

            motif_contig = _collapse_ranges(positions, binder_chain)
            num_fixed_positions = len(positions)

            key_res_payload = {str(k): [float(v[0]), v[1]] for k, v in key_res.items()}

            out_rows.append({
                "structure_id": sid,
                "pdb_name": meta.get("pdb_name") or sid,
                "pdb_path": str(merged_path),
                "binder_chain": binder_chain,
                "key_res": json.dumps(key_res_payload),
                "key_res_index": key_res_index,
                "fixed_structure": fixed_structure,
                "fixed_sequence": fixed_sequence,
                "fixed_sequence_pm": fixed_sequence_pm,
                "cdr_position": cdr_position,
                "motif_contig": motif_contig,
                "num_fixed_positions": num_fixed_positions,
            })

        out_csv = output_dir / "fixed_positions.csv"
        write_csv(out_csv, out_rows, [
            "structure_id",
            "pdb_name",
            "pdb_path",
            "binder_chain",
            "key_res",
            "key_res_index",
            "fixed_structure",
            "fixed_sequence",
            "fixed_sequence_pm",
            "cdr_position",
            "motif_contig",
            "num_fixed_positions",
        ])

    def write_manifest(self, ctx: StepContext) -> None:
        # Manifest is the fixed_positions.csv in output_dir
        return
