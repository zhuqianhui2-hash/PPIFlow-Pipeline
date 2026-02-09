# CLI Reference

This repo's README is intentionally "getting started" focused. This page is the reference.

## Commands

- `pipeline`: configure + execute (most common entry point)
- `configure`: write per-step configs + `steps.yaml` without running
- `execute`: run/resume from an existing `steps.yaml`
- `rank`: run ranking only (`rank_features` + `rank_finalize`)
- `wizard`: interactive setup (guided prompts)
- `orchestrate`: advanced multi-worker controller (this is the default controller used by `pipeline`/`execute`; use `--single-process` to opt out for debugging)

## Help

The definitive reference for flags is always:
```bash
python ppiflow.py <command> --help
```

## Common Flags

- `--output`: output directory (run identity)
- `--input`: YAML input (`design.yaml`)
- `--protocol`: `binder`, `vhh`, `antibody` (CLI-only mode)
- `--steps`: `all` or comma-separated step list
- `--output-mode`: `minimal` or `full`
- `--num-devices`: multi-GPU parallelism (`all`, or an integer)
- `--devices`: bind to specific GPUs (`0,2,3`) (implies `--num-devices 3`)
- `--num-rosetta-workers` / `--num-cpu-workers`: CPU worker pool size for Rosetta-heavy item steps
- `--single-process`: force legacy single-process execution (debug/regression only)

## Common Flags By Command

`pipeline` (configure + execute):
- Typical: `--input`, `--output`
- CLI-only mode: `--protocol`, `--name`, `--target_pdb`, `--target_chains`, plus protocol-specific fields (e.g. `--binder_length` or framework args)
- Runtime: `--num-devices`, `--devices`, `--num-rosetta-workers`, `--output-mode`, `--steps`, `--verbose`, `--continue-on-error`
- Config control: `--skip-config`, `--force-config`

`configure` (write configs only):
- Typical: `--input`, `--output`
- CLI-only input flags are accepted here too (to generate a YAML + configs without running)

`execute` (resume / run from an output directory):
- Typical: `--output`
- Runtime: `--steps`, `--num-devices`, `--devices`, `--num-rosetta-workers`, `--verbose`, `--continue-on-error`
- Recovery: `--retry-failed`, `--work-queue-rebuild`, `--steal-lock`

`rank` (ranking only):
- Typical: `--output`
- Note: `rank` is single-process only.

## YAML Input Schema (Minimal)

The README is YAML-first. Minimal schema highlights:

- Required: `protocol`, `name`, `target.pdb`, `target.chains`
- Binder: requires `binder.length`
- VHH: requires `framework.pdb`, `framework.heavy_chain`, `framework.cdr_length`
- Antibody: requires `framework.pdb`, `framework.heavy_chain`, `framework.light_chain`, `framework.cdr_length`
- Optional but common: `target.hotspots`, `sampling.samples_per_target`

Examples:

Binder:
```yaml
protocol: binder
name: my_binder
target:
  pdb: /path/to/target.pdb
  chains: ["A"]
  hotspots: ["A56"]
binder:
  length: "65-150"
sampling:
  samples_per_target: 200
```

VHH:
```yaml
protocol: vhh
name: my_vhh
target:
  pdb: /path/to/antigen.pdb
  chains: ["A"]
  hotspots: ["A56"]
framework:
  pdb: /path/to/framework.pdb
  heavy_chain: A
  cdr_length: "CDRH1,8-8,CDRH2,8-8,CDRH3,20-20"
sampling:
  samples_per_target: 200
```

Antibody:
```yaml
protocol: antibody
name: my_antibody
target:
  pdb: /path/to/antigen.pdb
  chains: ["A"]
  hotspots: ["A56"]
framework:
  pdb: /path/to/framework.pdb
  heavy_chain: A
  light_chain: B
  cdr_length: "CDRH1,8-8,CDRH2,8-8,CDRH3,12-12,CDRL1,11-11,CDRL2,3-3,CDRL3,9-9"
sampling:
  samples_per_target: 200
```

## Tool Wiring: Precedence and Env Vars

Tool paths can come from:
1. Your YAML `tools:` block (highest priority).
2. Environment variables (recommended via `source ./env.sh`).
3. Repo-local defaults under `assets/` created by `./install_ppiflow.sh`.

`env.sh` exports the common variables below (not every step needs all of them):
- `ROSETTA_BIN`, `ROSETTA_DB`, `ROSETTA_ENV`
- `AF3_WEIGHTS`, `AF3_DB_DIR`
- `AF3SCORE_REPO`, `AF3SCORE_ENV`, `AF3SCORE_PYTHON`, `AF3SCORE_CUDA_HOME`
- `FLOWPACKER_REPO`
- `PROTEINMPNN_REPO`, `MPNN_WEIGHTS`, `MPNN_SOLUBLE_WEIGHTS`
- `ABMPNN_WEIGHTS_DIR`, `ABMPNN_WEIGHTS_FILE` (and aliases)
- `DOCKQ_BIN`

See `documentation/pipeline_steps.md` for which steps require which tools.

## CLI-Only Mode (No YAML)

Binder required fields:
- `--protocol binder`
- `--name`
- `--target_pdb`
- `--target_chains`
- `--binder_length`
- `--output`

Antibody/VHH required fields add a framework:
- `--framework_pdb`
- `--heavy_chain`
- `--cdr_length`
- Antibody also needs `--light_chain`
