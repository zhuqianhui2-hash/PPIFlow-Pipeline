# Bundled Frameworks (Copy/Paste)

This repo bundles a small set of VHH (nanobody) and scFv antibody frameworks under `assets/frameworks/`.

You can use these with:
- CLI-only runs (pass the flags below to `python ppiflow.py pipeline ...`)
- YAML runs (set `framework.pdb`, `framework.heavy_chain`, `framework.light_chain` (antibody only), and `framework.cdr_length`)

Note on paths in YAML: framework PDB paths are resolved relative to the YAML file location. If your YAML lives in `examples/`, you likely want `../assets/frameworks/...`.

Chain ID note: `--heavy_chain` / `--light_chain` should match the chain IDs in the framework PDB file you provide (many scFv frameworks use `A/B`). During `configure`, PPIFlow rewrites framework chains to the pipeline's internal conventions (heavy `A`, light `C`).

## VHH (Nanobody) Frameworks (Single Chain `A`)

- 5JDS nanobody
  ```bash
  --framework_pdb assets/frameworks/5jds_nanobody_framework.pdb \
  --heavy_chain A \
  --cdr_length "CDRH1,8-8,CDRH2,8-8,CDRH3,21-21"
  ```

- 7EOW nanobody
  ```bash
  --framework_pdb assets/frameworks/7eow_nanobody_framework.pdb \
  --heavy_chain A \
  --cdr_length "CDRH1,8-8,CDRH2,8-8,CDRH3,20-20"
  ```

- 7XL0 nanobody
  ```bash
  --framework_pdb assets/frameworks/7xl0_nanobody_framework.pdb \
  --heavy_chain A \
  --cdr_length "CDRH1,8-8,CDRH2,7-7,CDRH3,15-15"
  ```

- 8COH nanobody
  ```bash
  --framework_pdb assets/frameworks/8coh_nanobody_framework.pdb \
  --heavy_chain A \
  --cdr_length "CDRH1,8-8,CDRH2,8-8,CDRH3,19-19"
  ```

- 8Z8V nanobody
  ```bash
  --framework_pdb assets/frameworks/8z8v_nanobody_framework.pdb \
  --heavy_chain A \
  --cdr_length "CDRH1,8-8,CDRH2,8-8,CDRH3,8-8"
  ```

## scFv / Antibody Frameworks (Heavy `A`, Light `B`)

- 6NOU scFv
  ```bash
  --framework_pdb assets/frameworks/6nou_scfv_framework.pdb \
  --heavy_chain A --light_chain B \
  --cdr_length "CDRH1,8-8,CDRH2,8-8,CDRH3,12-12,CDRL1,11-11,CDRL2,3-3,CDRL3,9-9"
  ```

- 6TCS scFv
  ```bash
  --framework_pdb assets/frameworks/6tcs_scfv_framework.pdb \
  --heavy_chain A --light_chain B \
  --cdr_length "CDRH1,9-9,CDRH2,7-7,CDRH3,14-14,CDRL1,10-10,CDRL2,3-3,CDRL3,9-9"
  ```

- 6ZQK scFv
  ```bash
  --framework_pdb assets/frameworks/6zqk_scfv_framework.pdb \
  --heavy_chain A --light_chain B \
  --cdr_length "CDRH1,8-8,CDRH2,8-8,CDRH3,13-13,CDRL1,6-6,CDRL2,3-3,CDRL3,9-9"
  ```
