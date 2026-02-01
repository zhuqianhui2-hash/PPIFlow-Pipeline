"""Quiet AF3Score featurisation logging (best-effort)."""

from __future__ import annotations


def _patch_featurisation() -> None:
    try:
        from alphafold3.data import featurisation
    except Exception:
        return

    original = getattr(featurisation, "featurise_input", None)
    if original is None:
        return

    def _quiet_featurise_input(*args, **kwargs):
        kwargs["verbose"] = False
        return original(*args, **kwargs)

    featurisation.featurise_input = _quiet_featurise_input


try:
    _patch_featurisation()
except Exception:
    # Never fail AF3Score if patching fails.
    pass
