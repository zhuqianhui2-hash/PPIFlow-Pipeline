from __future__ import annotations

from pathlib import Path
from typing import Any


def chain_sequences_from_pdb(pdb_path: str | Path, *, standard_only: bool = True) -> dict[str, str]:
    """
    Extract 1-letter amino-acid sequences per chain from a PDB.

    Notes:
    - Uses Biopython's PDBParser for correctness across common PDB variants.
    - Falls back to 'X' for residues that cannot be converted by seq1.
    - Only residues considered amino acids by Bio.PDB.is_aa() are included.
    """
    from Bio import PDB  # Biopython is already a dependency used elsewhere in the pipeline.
    from Bio.SeqUtils import seq1

    p = Path(pdb_path)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(p.stem, str(p))
    model = structure[0]

    seqs: dict[str, str] = {}
    for chain in model:
        residues: list[str] = []
        for res in chain:
            # Keep only amino-acid residues. If standard_only is enabled, we still preserve
            # non-standard amino acids as "X" rather than dropping them (prevents silent
            # sequence shortening when e.g. MSE appears in upstream PDBs).
            if not PDB.is_aa(res, standard=False):
                continue
            if standard_only and not PDB.is_aa(res, standard=True):
                residues.append("X")
                continue
            try:
                residues.append(seq1(res.get_resname()))
            except Exception:
                residues.append("X")
        seqs[str(chain.id)] = "".join(residues)
    return seqs


def get_chain_sequence_from_pdb(
    pdb_path: str | Path,
    chain_id: str,
    *,
    standard_only: bool = True,
) -> str | None:
    chain_id = str(chain_id or "").strip()
    if not chain_id:
        return None
    seqs = chain_sequences_from_pdb(pdb_path, standard_only=standard_only)
    return seqs.get(chain_id)


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text
