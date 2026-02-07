# PPIFlow (PPIFlow-Pipeline fork)

![](./assets/model.png)

PPIFlow is a flow-matching framework and unified design pipeline for de novo
protein binders. This fork supports binder, antibody, and VHH protocols with a
single resumable CLI pipeline.

## What You Need

- **GPU(s):** at least one CUDA GPU. Use `--num-devices` to run across multiple GPUs.
- **Disk:** minimal output mode (default) keeps only required artifacts; `--output-mode full` preserves heavy intermediates (plan for 10-50 GB per run depending on `samples_per_target`).
- **Files you must supply:**
  - A target PDB file
  - PPIFlow checkpoints (`binder.ckpt`, `antibody.ckpt`, `nanobody.ckpt`, `monomer.ckpt`)
  - AlphaFold 3 weights (`af3.bin.zst`)
  - (Optional) AlphaFold 3 database directory (for AF3Score inference)

## Key Concepts (2-Minute Primer)

- **Protocols:** `binder` designs a free-form binder chain; `antibody` designs an scFv (heavy + light chain); `vhh` designs a single-domain nanobody.
- **Chains and hotspots:** the target protein is "chain B" internally. Hotspots are residues on the target where contact is desired (e.g. `B3,B5-25,B72`).
- **`samples_per_target`:** how many backbone designs to generate. More samples = more candidates = longer runtime.
- **Pipeline steps at a glance:**

| Step | What it does |
|---|---|
| `gen` | Generate backbone conformations with flow matching |
| `seq1` | Round-1 sequence design (ProteinMPNN / AbMPNN) |
| `flowpacker1` | Side-chain packing (FlowPacker) |
| `af3score1` | Round-1 scoring (AlphaFold 3) |
| `rosetta_interface` | Rosetta interface energy analysis |
| `interface_enrich` | Extract enriched interface positions |
| `partial` | Partial-flow refinement |
| `seq2` | Round-2 sequence design |
| `flowpacker2` | Round-2 side-chain packing |
| `af3score2` | Round-2 scoring |
| `relax` | Rosetta relaxation |
| `rosetta_interface2` | Post-relax Rosetta interface analysis |
| `af3_refold` | AF3 refolding validation |
| `dockq` | DockQ structural quality |
| `rank_features` | Collect ranking features |
| `rank_finalize` | Produce ranked results |

## Quickstart

### 1) Install

Clone this repo:
```bash
git clone https://github.com/cytokineking/PPIFlow-Pipeline
```

Use the install script to set up environments, external tools, and paths:

```bash
./install_ppiflow.sh \
  --af3-weights-path /path/to/af3.bin.zst \
  --ppiflow-checkpoints-path /path/to/ppiflow_checkpoints \
  --install-os-deps \
  --install-conda \
  --write-env
```

The installer creates three conda environments:
- `ppiflow` (main pipeline)
- `ppiflow-af3score` (AF3Score + JAX)
- `ppiflow-rosetta` (Rosetta CLI)

<details>
<summary>Installer flags</summary>

| Flag | Default | What it does |
|---|---|---|
| `--af3-weights-path` | (required) | AF3 weights file |
| `--ppiflow-checkpoints-path` | (required) | Directory with PPIFlow checkpoints |
| `--af3-db-path` | (none) | AlphaFold 3 database directory |
| `--prefix` | repo root | Install root |
| `--pkg-manager` | `conda` | `conda` or `mamba` |
| `--skip-gpu-check` | false | Skip `nvidia-smi` detection |
| `--af3score-env-name` | `ppiflow-af3score` | Customize AF3Score env name |
| `--rosetta-env-name` | `ppiflow-rosetta` | Customize Rosetta env name |
| `--abmpnn-weights-path` | assets default | AbMPNN weights override |
| `--rosetta-db-path` | (auto) | Rosetta database path override |
| `--no-install-flowpacker` | false | Skip FlowPacker install |
| `--no-install-af3score` | false | Skip AF3Score install |
| `--no-install-dockq` | false | Skip DockQ install |

Run `./install_ppiflow.sh --help` for the full list.
</details>

After install, load tool paths and activate:
```bash
source ./env.sh
conda activate ppiflow
```

Manual option (manage tools yourself):
```bash
conda env create -f environment.yml
conda activate ppiflow
```

### 2) Recommended First Run: Interactive Wizard

```bash
python ppiflow.py wizard
```

The wizard prompts for: protocol, target PDB, target chains, hotspots, framework
(for antibody/VHH), samples per target, GPU selection, and optional advanced
filter thresholds. It writes `pipeline_input.json` and per-step configs under
the output directory, then runs the pipeline. Requires an interactive TTY.

The wizard does **not** configure advanced tool wiring (custom commands, scratch
directories, output pruning). For those, use a YAML input file or edit per-step
configs after `configure`.

### 3) Run with a YAML Input

Create `design.yaml` (see [Input Schema](#input-schema-designyaml)), then:

```bash
python ppiflow.py pipeline \
  --protocol binder \
  --preset fast \
  --input /path/to/design.yaml \
  --output /path/to/out_dir
```

### 4) Run with CLI-Only Inputs (No YAML)

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

The CLI writes `config/pipeline_input.yaml` and `pipeline_input.json` in the
output directory, then runs the pipeline.

## Pipeline CLI

### Commands

| Command | What it does |
|---|---|
| `pipeline` | Configure + execute (most common entry point) |
| `configure` | Write per-step configs + `steps.yaml` without running |
| `execute` | Run from existing `steps.yaml` (single-process by default; with `--num-devices` or `--num-rosetta-workers` routes to orchestrator) |
| `orchestrate` | Multi-worker orchestrator (advanced; called implicitly by `execute --num-devices/--num-rosetta-workers`) |
| `rank` | Run ranking only (`rank_features` + `rank_finalize`); single-process only |
| `wizard` | Interactive setup (guided prompts, requires TTY, zero flags) |

**Single-process `execute` vs multi-worker `orchestrate`:** When you pass `--num-devices` and/or `--num-rosetta-workers` to `pipeline` or `execute`, the CLI internally calls `orchestrate_pipeline()` which spawns worker pools. Only bare `execute` (without either flag) runs in single-process mode. You rarely need to call `orchestrate` directly.

The orchestrator writes artifacts under `.orchestrator/`:
- `.orchestrator/plan.json` — resolved execution plan and failure policy
- `.orchestrator/orchestrator.json` — run summary and status

**`rank` limitation:** The `rank` parser accepts `--num-devices` in `--help` output but rejects it at runtime. For multi-GPU ranking, use `execute --steps rank_features` or `orchestrate --steps rank_features`.

### Subcommand Flag Availability

Not all flags are available on every subcommand. Key differences:

| Flag group | `pipeline` | `configure` | `execute` | `rank` | `orchestrate` | `wizard` |
|---|---|---|---|---|---|---|
| Common args (protocol, preset, steps, etc.) | yes | yes | partial | partial | partial | -- |
| `--skip-config` / `--force-config` | yes | -- | -- | -- | -- | -- |
| CLI input args (target, binder, framework) | yes | yes | -- | -- | -- | -- |
| Work-queue flags | yes | yes | yes | yes | -- | -- |
| Run-lock flags | yes | -- | yes | yes | yes | -- |
| `--continue-on-error` | yes | yes† | yes | yes | -- | -- |
| `--verbose` | yes | yes† | yes | yes | -- | -- |
| `--num-devices` / `--devices` | yes | yes† | yes | parse-only* | yes | -- |
| Orchestrator pool overrides (`--num-rosetta-workers`) | yes | -- | yes | -- | yes | -- |
| Orchestrator-only flags | -- | -- | -- | -- | yes | -- |

\* `rank` accepts `--num-devices` in argparse but rejects it at runtime.
† `configure` accepts these flags via argparse, but most are operationally relevant only when running (`pipeline`/`execute`/`orchestrate`).

**Runtime constraints:**
- `pipeline --num-devices/--num-rosetta-workers` / `execute --num-devices/--num-rosetta-workers` rejects multi-step comma-lists for `--steps` (only `all` or a single step name allowed).
- `orchestrate --steps` accepts only a single step (no comma-lists).
- `orchestrate --configure` requires `--input` to be set.
- `orchestrate --no-bind` cannot be combined with `--num-devices`.

### Most-Used Flags

| Flag | Default | What it does |
|---|---|---|
| `--protocol` | (required) | `binder`, `antibody`, or `vhh` |
| `--preset` | `full` | `fast` (50 samples), `full` (200 samples), or `custom` |
| `--input` | (none) | Path to `design.yaml` |
| `--output` | (required) | Output directory |
| `--output-mode` | `minimal` | `minimal` or `full` (keep heavy intermediates) |
| `--steps` | `all` | Run specific steps: `all` or comma-separated list |
| `--num-devices` | (none) | Number of GPUs/workers (e.g. `4` or `all`) |
| `--devices` | (none) | Specific GPUs (e.g. `0,2,3`) |
| `--verbose` | false | Stream subprocess output to console |
| `--skip-config` | false | Reuse existing configs if present (auto-configures if missing) |
| `--force-config` | false | Always regenerate configs before running |
| `--continue-on-error` | false | Best-effort mode: continue to later steps even if a step fails, and tolerate per-item failures in work-queue steps. Downstream steps may run with reduced candidate sets. |

**Note on `--skip-config`:** this flag is best-effort. If config files are missing, the pipeline runs `configure` anyway regardless of `--skip-config`.

**Note on `--preset`:** presets set `sampling.samples_per_target` when that field is missing from the YAML. In CLI-only mode, `--samples_per_target` defaults to 100 and takes precedence over the preset default.

### Restart / Resume Safety

| Flag | Default | What it does |
|---|---|---|
| `--steal-lock` | false | Take over an output dir even if it appears locked by another process |
| `--run-lock-stale-seconds` | (derived) | Consider a lock stale after N seconds |
| `--no-run-lock` | false | Disable run lock entirely (debug only) |

### Work Queue Tuning

| Flag | Default | What it does |
|---|---|---|
| `--work-queue` | (see note) | Force-enable work queue (override YAML that disabled it). The work queue is enabled by default via runtime normalization; this flag is a force-enable override, not an on/off toggle. |
| `--work-queue-lease-seconds` | 300 | Lease duration per item |
| `--work-queue-max-attempts` | 2 | Max claims per item before it is marked failed |
| `--work-queue-batch-size` | 1 | Items per worker iteration |
| `--work-queue-leader-timeout` | 600 | Timeout for leader-mode steps |
| `--work-queue-wait-timeout` | (none) | Max wait for items to appear |
| `--retry-failed` | false | Reset failed/blocked/running items for re-attempt |
| `--work-queue-reuse` | false | Accept legacy outputs that lack `step_meta.json` |
| `--work-queue-strict` | false | Disable reuse; enforce strict completion checks |
| `--work-queue-rebuild` | false | Drop and rebuild `queue.db` from outputs on disk |

**Retry knob differences:** `--work-queue-max-attempts` controls how many times a single *work item* can be claimed before it is permanently failed. `--max-retries` (orchestrator-only) controls how many times the orchestrator restarts a *step's entire worker pool*. These are independent knobs at different levels.

### Orchestrator-Only Flags

| Flag | Default | What it does |
|---|---|---|
| `--configure` | false | Run `configure` if `steps.yaml` is missing (requires `--input`) |
| `--pool-size` | (auto) | Override per-step worker pool size |
| `--max-retries` | (none) | Max times the orchestrator restarts a step's worker pool |
| `--failure-policy` | (none) | `allow` (keep going), `strict` (stop on failure), `threshold` |
| `--no-bind` | false | Don't set `CUDA_VISIBLE_DEVICES` (cannot combine with `--num-devices`) |

### Orchestrator Pool Overrides (Advanced)

These flags are accepted by `pipeline`, `execute`, and `orchestrate` to control pool sizing for CPU-heavy (Rosetta) item steps.

| Flag | Default | What it does |
|---|---|---|
| `--num-rosetta-workers` | (auto) | Override pool size for Rosetta (CPU) item steps |
| `--num-cpu-workers` | (alias) | Alias for `--num-rosetta-workers` |

### CLI-Only Input Flags (When `--input` Is Omitted)

**Required:**
- `--protocol`, `--name`
- `--target_pdb`, `--target_chains`
- Binder: `--binder_length`
- Antibody/VHH: `--framework_pdb`, `--heavy_chain`, `--cdr_length` (plus `--light_chain` for antibody)

**Optional:**
- `--hotspots`
- `--samples_per_target` (default 100)
- `--ppiflow_ckpt` (required if not provided by install/env)

If you used the installer and `source ./env.sh`, most tool paths can be omitted.

### Design Knobs

| Flag | Default | What it does |
|---|---|---|
| `--seq1_num_per_backbone` | binder: 16, ab/vhh: 8 | Sequences per backbone (round 1) |
| `--seq1_temp` | binder: 0.2, ab/vhh: 0.5 | Sampling temperature (round 1) |
| `--seq1_bias_large_residues` | binder: true, ab/vhh: false | Bias bulky residues |
| `--seq1_bias_num` | binder: 8, ab/vhh: 0 | Number of biased sequences |
| `--seq1_bias_residues` | `F,M,W` (binder) | Residues to bias |
| `--seq1_bias_weight` | 0.7 (binder) | Bias weight |
| `--seq2_num_per_backbone` | 4 | Sequences per backbone (round 2) |
| `--seq2_temp` | 0.1 | Sampling temperature (round 2) |
| `--seq2_use_soluble_ckpt` | true | Use soluble checkpoint for round 2 |
| `--partial_start_t` | 0.6 | Partial-flow start time |
| `--partial_samples_per_target` | 8 | Partial-flow samples per target |

Antibody/VHH convenience aliases:

| Flag | Maps to |
|---|---|
| `--vhh_backbones` | `--samples_per_target` |
| `--vhh_cdr1_num` | `--seq1_num_per_backbone` |
| `--vhh_partial_num` | `--partial_samples_per_target` |
| `--vhh_cdr2_num` | `--seq2_num_per_backbone` |

### Filters / Ranking

| Flag | Default | What it does |
|---|---|---|
| `--af3score1_iptm_min` | 0.2 | AF3Score round-1 ipTM filter |
| `--af3score1_ptm_min` | binder: 0.2, ab/vhh: None | AF3Score round-1 pTM filter |
| `--af3score1_top_k` | None | Keep top K after round-1 scoring |
| `--af3score2_iptm_min` | 0.5 | AF3Score round-2 ipTM filter |
| `--af3score2_ptm_min` | 0.8 | AF3Score round-2 pTM filter |
| `--af3score2_top_k` | None | Keep top K after round-2 scoring |
| `--af3refold_iptm_min` | 0.7 | AF3 refold ipTM filter |
| `--af3refold_ptm_min` | 0.8 | AF3 refold pTM filter |
| `--af3refold_dockq_min` | 0.49 | AF3 refold DockQ filter |
| `--af3refold_num_samples` | 5 | Refold samples per seed |
| `--af3refold_model_seeds` | `0-19` | Refold model seeds |
| `--af3refold_no_templates` | true | Disable templates for refold |
| `--interface_energy_min` | -5.0 | Rosetta interface energy cutoff (REU) |
| `--interface_distance` | 10.0 | Rosetta interface distance cutoff (A) |
| `--relax_max_iter` | 170 | Rosetta relax max iterations |
| `--relax_fixbb` | false | Fix backbone during relax |
| `--fixed_chains` | (none) | Chains to fix backbone (e.g. `A_B`) |
| `--dockq_min` | 0.49 | DockQ minimum threshold |
| `--rank_top_k` | 30 | Top K to keep in ranking |
| `--af3-num-workers` | (auto) | AF3Score worker count |

### Tool Paths

| Flag | What it points to |
|---|---|
| `--ppiflow_ckpt` | PPIFlow checkpoint |
| `--mpnn_ckpt` | ProteinMPNN weights |
| `--mpnn_ckpt_soluble` | ProteinMPNN soluble weights |
| `--abmpnn_ckpt` | AbMPNN weights |
| `--af3score_repo` | AF3Score repo |
| `--flowpacker_repo` | FlowPacker repo |
| `--af3_weights` | AF3 weights file |
| `--mpnn_repo` | ProteinMPNN repo |
| `--abmpnn_repo` | AbMPNN repo |
| `--mpnn_run` | Path to `protein_mpnn_run.py` |
| `--abmpnn_run` | Path to `protein_mpnn_run.py` (AbMPNN) |
| `--rosetta_bin` | `rosetta_scripts` binary |
| `--rosetta_db` | Rosetta database |
| `--dockq_bin` | DockQ binary |

## Input Schema (`design.yaml`)

Example templates (these are schema-valid templates; fill in real paths before use):

- [assets/examples/binder_minimal.yaml](assets/examples/binder_minimal.yaml)
- [assets/examples/antibody_minimal.yaml](assets/examples/antibody_minimal.yaml)
- [assets/examples/vhh_minimal.yaml](assets/examples/vhh_minimal.yaml)

Validation highlights:

- `protocol`, `name`, `target.pdb`, `target.chains` are required.
- Binder: `binder.length` required; no `framework` block.
- Antibody: `framework.pdb`, `framework.heavy_chain`, `framework.light_chain`, `framework.cdr_length` required.
- VHH: `framework.pdb`, `framework.heavy_chain`, `framework.cdr_length` required; omit `framework.light_chain`.

## Antibody & VHH Framework Quickstart

Bundled frameworks live in `assets/frameworks/`. Use the following copy-paste flag blocks.

<details>
<summary>VHH (nanobody) frameworks (single-chain A)</summary>

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
</details>

<details>
<summary>scFv/antibody frameworks (heavy A, light B)</summary>

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
</details>

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
  - `C` -> whole chain C is hotspot
  - `C62` -> single residue
  - `C62-73` -> range on chain C
  - Mixed list: `A3,A5-25,B72,B75`

Behavior (pipeline runs):

- **All protocols** accept multi-chain targets via `target.chains`. During
  `configure`, all target chains are concatenated into a single gapped **chain B** PDB
  written under `inputs/` (at the run root, alongside `config/`, `output/`, etc.).
- **Chain conventions**:
  - Binder/VHH: binder is **A**, target is **B**
  - Antibody: heavy **A**, light **C**, target **B**
  Framework PDBs are rewritten under `inputs/` to enforce this, and the
  binder chain is forced to **A** in pipeline inputs.
- **Hotspots** are specified on the original target PDB and then mapped onto
  concatenated chain **B**. The concatenated target uses fresh integer residue
  numbering; mapping is recorded in `inputs/target_chain_map.json`.

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
- `config/step_rosetta_interface2.yaml` (post-relax Rosetta interface; auto-generated)
- `config/step_dockq.yaml` (optional)
- `config/step_af3_refold.yaml` (optional)

Each step config supports a `command` field that can be a string or list, and
will be executed as a subprocess during `execute`.

## Output Layout

The pipeline writes a deterministic layout under your output directory:

- `config/`: per-step YAMLs generated by `configure`
- `inputs/`: derived inputs written during `configure` (e.g. concatenated target PDB, chain maps, renamed framework PDBs, mapped-hotspots file)
- `steps.yaml`: manifest of pipeline steps
- `pipeline_input.json`: normalized input (resume identity)
- `pipeline_state.json`: run metadata + tool stamps
- `output/`: step outputs (backbones, seqs, scores, rosetta, etc.)
- `results/`: ranked outputs (rank step; reruns may create `results_vN/` at the run root)
- `.work/<step>/queue.db`: progress ledger for resume
- `.ppiflow_lock/`: run lock directory (prevents concurrent writes)
- `.orchestrator/`: orchestrator plan and status (when using multi-GPU)

### Step Name vs Output Directory

Some step names don't match their output directory. The most confusing mappings:

| Step name | Output directory |
|---|---|
| `partial` | `output/partial_flow` |
| `rank_features` | `results/features` |
| `rank_finalize` | `results/` |
| `rosetta_interface2` | `output/rosetta_interface2` |

### Output Mode and Optional Files

Minimal mode (default) keeps only artifacts needed for resume and ranking. AF3 CIFs are always kept in `output/af3score_round*/cif/` and `output/af3_refold/cif/`.

`--output-mode full` or `output.keep_optional` keys preserve additional intermediates under `output/_optional/`. Common optional keys (step-dependent; see `src/pipeline/prune.py` for the definitive list):

| Key | Steps that produce it |
|---|---|
| `after_pdbs` | flowpacker1, flowpacker2 |
| `flowpacker_outputs` | flowpacker1, flowpacker2 |
| `af3score_outputs` | af3score1, af3score2, af3_refold |
| `af3_input_batch` | af3score1, af3score2, af3_refold |
| `single_chain_cif` | af3score1, af3score2, af3_refold |
| `json` | af3score1, af3score2, af3_refold |
| `pdbs` | af3score1, af3score2 |
| `af3score_subprocess_logs` | af3score1, af3score2, af3_refold |
| `rosetta_jobs` | rosetta_interface, relax |
| `wandb`, `yaml`, `config.yaml` | partial |
| `.tmp` | most steps |

- `scratch/`: transient work dirs for large intermediates; safe to delete after the run unless you override `output.scratch_dir`.
- `logs/`: step logs (kept unless `output.keep_logs` is false).

### Heartbeat / Status Files

- `status.json` and `status_rank*.json` written to the output root.
- Set `PPIFLOW_HEARTBEAT=0` to disable. Interval: `PPIFLOW_HEARTBEAT_INTERVAL` (default 30s).

## Parallelization, Resume, and Run Locking

PPIFlow runs on a single GPU or many GPUs. To scale out, set `--num-devices`
(one worker per GPU). You can stop and resume later with a different number of
GPUs.

### Common Workflows

```bash
# Full pipeline on all GPUs
python ppiflow.py pipeline --protocol binder --input design.yaml \
  --output out_dir --num-devices all

# Resume after interruption (any GPU count)
python ppiflow.py execute --output out_dir --num-devices all

# Run a single step on multiple GPUs
python ppiflow.py execute --output out_dir --steps af3score1 --num-devices 4

# Run only ranking (single-process)
python ppiflow.py rank --output out_dir

# Retry failed items only
python ppiflow.py execute --output out_dir --retry-failed

# Rebuild progress DB from outputs (if .work/ is missing or corrupt)
python ppiflow.py execute --output out_dir --work-queue-rebuild

# Best-effort completion (continue despite failures)
python ppiflow.py execute --output out_dir --continue-on-error

# Change GPU count between runs
python ppiflow.py execute --output out_dir --num-devices 2
```

### Restart Decision Tree

**"I want to resume the same run"**
Rerun with the same `--output` directory:
```bash
python ppiflow.py execute --output out_dir
# or with multi-GPU:
python ppiflow.py execute --output out_dir --num-devices all
```

**"It says the output is locked"**
The pipeline writes a lock under `.ppiflow_lock/` to prevent two controllers from writing to the same output directory simultaneously.

- **Wait:** the lock has a staleness timeout (derived from heartbeat interval). If the old job crashed, the lock will expire and your retry will succeed.
- **Force takeover:** if you're sure the old job is dead:
  ```bash
  python ppiflow.py execute --output out_dir --steal-lock
  ```
- **Custom staleness:** `--run-lock-stale-seconds 120` (treat lock as stale after 2 minutes).
- **Disable entirely (debug only):** `--no-run-lock`

**"The run is stuck after restart"**
1. Check which step is stuck: look at logs under `logs/` or the `PPIFLOW_STAGE` in status files.
2. If it's a work-queue step, confirm `.work/<step>/queue.db` exists.
3. Knobs:
   - `--work-queue-rebuild`: reconstruct `queue.db` from outputs on disk (use when `.work/` is missing or corrupt).
   - `--retry-failed`: reset failed/blocked/running items so they can be reattempted.
   - `--work-queue-lease-seconds 60`: shorten the lease window in preemptible environments.
   - `--work-queue-max-attempts 5`: allow more re-attempts per item after preemption.

**"I changed the design input or tool paths and now it errors"**
`pipeline_state.json` enforces strict identity:
- Input hash mismatch -> use a new output directory.
- Tool path or checksum mismatch -> use a new output directory.

See `docs/real_world_resume_restart_todo.md` for detailed failure-mode recovery patterns.

### Resume Notes

- Resume uses preserved outputs as the source of truth. If `.work/` is missing, the work queue is rebuilt from outputs.
- Strict completion prefers output metadata (input/config/tool stamps). For legacy outputs without metadata, use `--work-queue-reuse` to opt in.

## Troubleshooting

- **`steps.yaml not found`**: run `configure` (or `pipeline`) first.
- **`pipeline_state.json ... mismatch`**: change output directory or ensure tool paths and inputs match the original run.
- **Lock errors**: see [Restart Decision Tree](#restart-decision-tree).
- **External tools failing**: verify paths in `tools` or CLI flags, and confirm per-step `command` entries are correct. Check `logs/` for files like `01_ppiflow_rank0.log` or `04_rosetta_interface_rank0.log` (the exact filename is also printed as `log:` when each step starts).
- **Where to find logs**: step logs are written to `logs/{idx:02d}_{step.name}{_rankX}.log` (rank suffix appears when the work queue is enabled). Use `--verbose` to also stream to console.

## Appendix: Legacy Scripts

Legacy demos and shell pipelines from the original PPIFlow repo live under `legacy_files/`, and the original entrypoint scripts remain in `src/entrypoints/`. These are not the recommended path for end-to-end usage but are kept for reference.

## License

The original PPIFlow code is licensed under an Attribution-NonCommercial-ShareAlike 4.0 International license. This pipeline is licensed under the same license to respect the upstream repository.
