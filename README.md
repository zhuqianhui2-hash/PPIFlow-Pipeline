# PPIFlow (PPIFlow-Pipeline fork)

![](./assets/model.png)

PPIFlow is a flow-matching framework and unified design pipeline for de novo
protein binders. This fork supports binder, antibody, and VHH protocols with a
single resumable CLI pipeline.

## Quickstart (Unified Pipeline)

### 1) Install

Clone this repo:
```bash
git clone https://github.com/cytokineking/PPIFlow-Pipeline
```

Recommended: use the install script to set up the environment, external tool
repos, and paths in one place. This fork expects you to supply:
- AF3 weights (`af3.bin.zst`): Obtain from Google
- PPIFlow checkpoints (`binder.ckpt`, `antibody.ckpt`, `nanobody.ckpt`, `monomer.ckpt`): Download from https://drive.google.com/drive/folders/1BcIBUL2yq1gOchHfN68-AcZK3hiMAMVN?usp=drive_link

```bash
./install_ppiflow.sh \
  --af3-weights-path /path/to/af3.bin.zst \
  --ppiflow-checkpoints-path /path/to/ppiflow_checkpoints \
  --install-os-deps \
  --install-conda \
  --write-env
```

The installer creates:
- `ppiflow` env (main pipeline)
- `ppiflow-af3score` env (AF3Score + JAX)
- `ppiflow-rosetta` env (Rosetta CLI)

Optional flags:
- `--no-install-flowpacker` or `--no-install-af3score` to skip those tools
- `--no-install-dockq` to skip DockQ
- `--rosetta-db-path /path/to/rosetta/database` to override Rosetta DB

Note: the full pipeline cannot be run without installing all tools.

After install, load tool paths and activate the main environment:
```bash
source ./env.sh
conda activate ppiflow
```
If you don’t want to activate the env, use the full path:
```bash
/path/to/PPIFlow-Pipeline/.miniforge3/envs/ppiflow/bin/python ppiflow.py ...
```

Manual option (if you want to manage tools yourself):

```bash
conda env create -f environment.yml
conda activate ppiflow
```

You will still need paths to external tools (FlowPacker, ProteinMPNN or
AbMPNN, AF3Score/AF3) and PPIFlow checkpoints. Rosetta CLI (`rosetta_scripts`)
is installed into a separate `ppiflow-rosetta` conda environment by the installer.

### 2) Run with a YAML input

Create `design.yaml` (see schema below), then run:

```bash
python ppiflow.py pipeline \
  --protocol binder \
  --preset fast \
  --input /path/to/design.yaml \
  --output /path/to/out_dir
```

### 2b) Run with the interactive wizard

```bash
python ppiflow.py wizard
```

### 3) Run with CLI-only inputs (no YAML)

```bash
python ppiflow.py pipeline \
  --protocol binder \
  --name il7ra_001 \
  --target_pdb /path/to/target.pdb \
  --target_chains B \
  --hotspots B3,B5-25,B72,B75 \
  --binder_length 75-90 \
  --samples_per_target 100 \
  --ppiflow_ckpt /path/to/binder.ckpt \
  --output /path/to/out_dir
```

The CLI will write `config/pipeline_input.yaml` and `pipeline_input.json` in the
output directory and then run the pipeline.

## Pipeline CLI

The unified CLI lives in `ppiflow.py` and exposes these commands:

- `pipeline`: configure + execute
- `configure`: write per-step configs + `steps.yaml`
- `execute`: run from `steps.yaml`
- `rank`: run the rank step only
- `orchestrate`: run per-step pools (advanced)
- `wizard`: interactive setup (guided prompts)

Common flags:

- `--protocol`: `binder|antibody|vhh` (required unless input YAML already has it)
- `--preset`: `fast|full|custom` (default: `full`)
- `--input`: path to `design.yaml` (optional)
- `--output`: output directory (required)
- `--steps`: `all|gen,seq1,flowpacker1,af3score1,rosetta_interface,interface_enrich,partial,seq2,flowpacker2,af3score2,relax,af3_refold,dockq,rank_features,rank_finalize`
- `--reuse`: skip outputs that already exist
- `--work-queue`: enable parallel/resume engine (default in configs)
- `--num-devices`: number of GPUs/workers (e.g. `4` or `all`). For specific GPUs: `--num-devices 3 --devices 0,2,3`
- `--skip-config`: reuse existing configs if present (auto-configure if missing)
- `--force-config`: always regenerate configs before running

Notes:

- `rank` command runs `rank_features` + `rank_finalize`.
- `--continue-on-error` allows a step to complete even if some items fail.

CLI-only input flags (when `--input` is omitted):

- **Required (no YAML):**
  - `--protocol`, `--name`
  - `--target_pdb`, `--target_chains`
  - Binder: `--binder_length`
  - Antibody/VHH: `--framework_pdb`, `--heavy_chain`, `--cdr_length` (plus `--light_chain` for antibody)
- **Optional:**
  - `--hotspots`
  - `--samples_per_target` (default 100)
  - `--ppiflow_ckpt` (required if not provided by your install/env)

If you used the installer and `source ./env.sh`, most tool paths can be omitted.

Advanced knobs (optional):

- Work-queue tuning: `--work-queue-max-attempts`, `--retry-failed`, `--work-queue-strict`,
  `--work-queue-rebuild`, `--work-queue-lease-seconds`, `--work-queue-wait-timeout`
- Filters: `--af3score1_iptm_min`, `--af3score2_iptm_min`, `--af3score2_ptm_min`,
  `--dockq_min`
- Rosetta: `--interface_energy_min`, `--interface_distance`, `--relax_max_iter`,
  `--relax_fixbb`, `--fixed_chains`
- Ranking: `--rank_top_k`
- GPU binding (advanced): `--devices 0,1,2` or `--devices all`
- AF3Score: `--af3-num-workers` (defaults to run_af3score.py default)
- Tool paths: `--ppiflow_ckpt`, `--mpnn_ckpt`, `--abmpnn_ckpt`, `--af3score_repo`,
  `--flowpacker_repo`, `--af3_weights`, `--mpnn_repo`, `--rosetta_bin`,
  `--rosetta_db`, `--abmpnn_repo`, `--mpnn_run`, `--abmpnn_run`, `--dockq_bin`
- Sequence design knobs: `--seq1_num_per_backbone`, `--seq1_temp`,
  `--seq1_bias_large_residues`, `--seq1_bias_num`, `--seq2_num_per_backbone`,
  `--seq2_temp`, `--seq2_use_soluble_ckpt`

## Input Schema (`design.yaml`)

Example templates:

- [assets/examples/binder_minimal.yaml](assets/examples/binder_minimal.yaml)
- [assets/examples/antibody_minimal.yaml](assets/examples/antibody_minimal.yaml)
- [assets/examples/vhh_minimal.yaml](assets/examples/vhh_minimal.yaml)

Validation highlights:

- `protocol`, `name`, `target.pdb`, `target.chains` are required.
- Binder: `binder.length` required; no `framework` block.
- Antibody: `framework.pdb`, `framework.heavy_chain`, `framework.light_chain`, `framework.cdr_length` required.
- VHH: `framework.pdb`, `framework.heavy_chain`, `framework.cdr_length` required; omit `framework.light_chain`.

## Antibody & VHH Framework Quickstart

Bundled frameworks live in `assets/frameworks/`. Use the following copy‑paste flag blocks.

VHH (nanobody) frameworks (single‑chain A):

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

scFv/antibody frameworks (heavy A, light B):

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

Full runnable examples:

```bash
python ppiflow.py pipeline \
  --protocol vhh \
  --name vhh_demo \
  --target_pdb /path/to/antigen.pdb \
  --target_chains C \
  --framework_pdb assets/frameworks/7eow_nanobody_framework.pdb \
  --heavy_chain A \
  --cdr_length "CDRH1,8-8,CDRH2,8-8,CDRH3,20-20" \
  --samples_per_target 50 \
  --output /path/to/out_dir
```

```bash
python ppiflow.py pipeline \
  --protocol antibody \
  --name ab_demo \
  --target_pdb /path/to/antigen.pdb \
  --target_chains C \
  --framework_pdb assets/frameworks/6nou_scfv_framework.pdb \
  --heavy_chain A --light_chain B \
  --cdr_length "CDRH1,8-8,CDRH2,8-8,CDRH3,12-12,CDRL1,11-11,CDRL2,3-3,CDRL3,9-9" \
  --samples_per_target 50 \
  --output /path/to/out_dir
```

## Multi-Chain Targets, Concatenation, and Chain Conventions

Supported hotspot specs (ColabDesign/BindCraft-style):

- Chain ID is always required:
  - `C` → whole chain C is hotspot
  - `C62` → single residue
  - `C62-73` → range on chain C
  - Mixed list: `A3,A5-25,B72,B75`

Behavior (pipeline runs):

- **All protocols** accept multi-chain targets via `target.chains`. During
  `configure`, all target chains are concatenated into a single gapped **chain B** PDB
  written under `output/inputs/`.
- **Chain conventions**:
  - Binder/VHH: binder is **A**, target is **B**
  - Antibody: heavy **A**, light **C**, target **B**
  Framework PDBs are rewritten under `output/inputs/` to enforce this, and the
  binder chain is forced to **A** in pipeline inputs.
- **Hotspots** are specified on the original target PDB and then mapped onto
  concatenated chain **B**. The concatenated target uses fresh integer residue
  numbering; mapping is recorded in `output/inputs/target_chain_map.json`.

See `docs/target_chain_concatenation_spec.md` for details.

## External Tool Wiring

The unified pipeline orchestrates multiple external tools. The pipeline engine
runs PPIFlow generation, Rosetta interface/relax, and partial-flow steps
directly. Sequence design can run automatically if `mpnn_run` or `abmpnn_run`
is provided; FlowPacker, AF3Score, and DockQ are executed through external
commands.

After `configure`, edit these per-step configs to provide commands:

- `config/step_seq1.yaml`
- `config/step_flowpacker1.yaml`
- `config/step_af3score1.yaml`
- `config/step_rosetta_interface.yaml` (Rosetta CLI runs automatically)
- `config/step_interface_enrich.yaml` (uses Rosetta interface energies)
- `config/step_seq2.yaml`
- `config/step_flowpacker2.yaml`
- `config/step_af3score2.yaml`
- `config/step_relax.yaml` (Rosetta CLI runs automatically)
- `config/step_dockq.yaml` (optional)
- `config/step_af3_refold.yaml` (optional)

Each step config supports a `command` field that can be a string or list, and
will be executed as a subprocess during `execute`.

## Output Layout

The pipeline writes a deterministic layout under your output directory:

- `config/`: per-step YAMLs generated by `configure`
- `steps.yaml`: manifest of pipeline steps
- `pipeline_input.json`: normalized input (resume identity)
- `pipeline_state.json`: run metadata + tool stamps
- `output/`: step outputs (backbones, seqs, scores, rosetta, etc.)
- `results/`: ranked outputs (rank step; reruns may create `results_vN/`)
- `.work/<step>/queue.db`: progress ledger for resume

Heartbeat/status files (optional):

- `status.json` and `status_rank*.json` written to the output root
- set `PPIFLOW_HEARTBEAT=0` to disable

## Parallelization & Resume

PPIFlow runs on a single GPU or many GPUs. To scale out, set `--num-devices`
(one worker per GPU). You can stop and resume later with a different number of
GPUs. Defaults are tuned for large design batches where some failures are
expected.

Common patterns:

```bash
# Parallelize across multiple GPUs (one worker per GPU)
python ppiflow.py execute --output out_dir --num-devices all

# Resume after interruption (any GPU count)
python ppiflow.py execute --output out_dir --num-devices all

# Retry failed items only
python ppiflow.py execute --output out_dir --retry-failed

# Strict mode: verify outputs for completed items (fails if missing)
python ppiflow.py execute --output out_dir --work-queue-strict

# If the progress DB is missing or corrupted, rebuild from outputs
python ppiflow.py execute --output out_dir --work-queue-rebuild
```

## Troubleshooting

- `steps.yaml not found`: run `configure` (or `pipeline`) first.
- `pipeline_state.json ... mismatch`: change output directory or ensure tool paths
  and inputs match the original run.
- External tools failing: verify paths in `tools` or CLI flags, and confirm
  per-step `command` entries are correct.

## Appendix: Legacy Scripts (Short)

Legacy demos and shell pipelines from the original PPIFlow repo live under `legacy/`, and the original entrypoint scripts remain in `src/entrypoints/`. These are not the recommended path for end-to-end usage but are kept for reference.

## License

The original PPIFlow code is licensed under an Attribution-NonCommercial-ShareAlike 4.0 International license. This pipeline is licensed under the same license to respect the upstream repository.
