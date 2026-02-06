from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
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


def files_identical(src: Path, dst: Path, *, chunk_size: int = 1024 * 1024) -> bool:
    """Return True only if src and dst are byte-identical."""
    try:
        if os.path.samefile(src, dst):
            return True
    except Exception:
        pass
    try:
        src_stat = src.stat()
        dst_stat = dst.stat()
    except FileNotFoundError:
        return False
    except Exception:
        src_stat = None
        dst_stat = None
    if src_stat is not None and dst_stat is not None:
        if src_stat.st_size != dst_stat.st_size:
            return False
        # If they are the same inode, they are the same bytes.
        try:
            if (src_stat.st_dev, src_stat.st_ino) == (dst_stat.st_dev, dst_stat.st_ino):
                return True
        except Exception:
            pass
    try:
        with src.open("rb") as s, dst.open("rb") as d:
            while True:
                sb = s.read(chunk_size)
                db = d.read(chunk_size)
                if sb != db:
                    return False
                if not sb:
                    return True
    except FileNotFoundError:
        return False


def promote_file(src: Path, dst: Path, *, allow_reuse: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if files_identical(src, dst):
            return
        # Destination exists but differs from source; do not silently keep stale outputs.
        raise RuntimeError(f"collision at {dst}")
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def promote_file_atomic(src: Path, dst: Path, *, allow_reuse: bool) -> None:
    """Promote a file into its final destination atomically.

    If possible, use a hardlink for speed; otherwise copy into a temp file in the
    destination directory and then atomically install it into place. This avoids
    leaving partially-written destination files (common on cross-filesystem copies).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if files_identical(src, dst):
            return
        raise RuntimeError(f"collision at {dst}")
    tmp = dst.parent / f".tmp.{dst.name}.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        try:
            os.link(src, tmp)
        except Exception:
            shutil.copy2(src, tmp)
        # Prefer linking the fully-written tmp into place so we never overwrite an existing dst.
        # If hardlinks aren't supported, fall back to os.replace().
        try:
            os.link(tmp, dst)
        except FileExistsError:
            if dst.exists() and files_identical(src, dst):
                return
            raise RuntimeError(f"collision at {dst}")
        except OSError:
            if dst.exists():
                if files_identical(src, dst):
                    return
                raise RuntimeError(f"collision at {dst}")
            os.replace(tmp, dst)
    finally:
        # Best-effort cleanup of the temp path.
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def promote_tree(src: Path, dst: Path, *, allow_reuse: bool) -> None:
    if not src.exists():
        return
    for root, _, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        for fname in files:
            promote_file_atomic(root_path / fname, dst / rel / fname, allow_reuse=allow_reuse)
