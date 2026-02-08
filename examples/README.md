# Examples

This folder contains runnable example jobs that should work out of the box after you run `./install_ppiflow.sh` and activate the environment:

```bash
source ./env.sh
conda activate ppiflow
```

## Bundled Target: `examples/targets/PDL1.pdb`

We bundle `PDL1.pdb` as a small test target so you can verify that installation, tool wiring, and the pipeline controller all work end-to-end.

Provenance: this file is copied byte-for-byte from FreeBindCraft (`FreeBindCraft/example/PDL1.pdb`).

## Run The Examples

Binder:
```bash
python ppiflow.py pipeline --input examples/pdl1_binder.yaml --output runs/example_pdl1_binder
```

VHH (nanobody):
```bash
python ppiflow.py pipeline --input examples/pdl1_vhh.yaml --output runs/example_pdl1_vhh
```

Antibody (scFv):
```bash
python ppiflow.py pipeline --input examples/pdl1_antibody.yaml --output runs/example_pdl1_antibody
```

These example configs are intentionally small (`samples_per_target: 10`). They are for validating that the pipeline runs, not for producing high-quality designs.

