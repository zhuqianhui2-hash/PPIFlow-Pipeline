from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Iterable


def safe_id_from_relpath(relpath: str, *, max_len: int = 120) -> str:
    raw = str(relpath)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    if not safe:
        safe = hashlib.sha256(raw.encode()).hexdigest()[:16]
    if len(safe) > max_len:
        h = hashlib.sha256(raw.encode()).hexdigest()[:8]
        safe = safe[: max_len - (len(h) + 2)] + "__" + h
    return safe


def compute_run_stems(pdbs: Iterable[Path], input_root: Path) -> dict[Path, str]:
    pdbs = list(pdbs)
    counts: dict[str, int] = {}
    for p in pdbs:
        counts[p.stem] = counts.get(p.stem, 0) + 1
    duplicates = {stem for stem, count in counts.items() if count > 1}
    mapping: dict[Path, str] = {}
    for pdb in pdbs:
        if pdb.stem in duplicates:
            rel = pdb.relative_to(input_root).as_posix()
            safe_id = safe_id_from_relpath(rel)
            mapping[pdb] = f"{safe_id}__{pdb.stem}"
        else:
            mapping[pdb] = pdb.stem
    return mapping


def file_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def promote_file(src: Path, dst: Path, *, allow_reuse: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not allow_reuse:
            if file_hash(src) != file_hash(dst):
                raise RuntimeError(f"collision at {dst}")
            return
        if src.stat().st_size == dst.stat().st_size:
            return
        if file_hash(src) != file_hash(dst):
            raise RuntimeError(f"collision at {dst}")
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def promote_tree(src: Path, dst: Path, *, allow_reuse: bool) -> None:
    if not src.exists():
        return
    for root, _, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        for fname in files:
            promote_file(root_path / fname, dst / rel / fname, allow_reuse=allow_reuse)
