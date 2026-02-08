# PPIFlow (PPIFlow-Pipeline fork)

![](./assets/model.png)

PPIFlow is a unified, resumable CLI pipeline for de novo protein binders. This fork supports binder, antibody, and VHH protocols.

## What You Need

- **GPU(s):** at least one CUDA GPU. Use `--num-devices` to scale across multiple GPUs.
- **Files you must supply:**
  - A target PDB file
  - PPIFlow checkpoints (`binder.ckpt`, `antibody.ckpt`, `nanobody.ckpt`, `monomer.ckpt`): see the upstream PPIFlow release instructions
  - AlphaFold 3 weights (`af3.bin.zst`): follow the AlphaFold 3 / AF3Score instructions for obtaining model weights
  - (Optional) AlphaFold 3 database directory (for AF3Score inference)

## What The Pipeline Runs (16 Steps)

<details>
<summary>Show the 16 pipeline steps</summary>

| Step | What it does |
|---|---|
| `gen` | Generate backbone conformations with flow matching |
| `seq1` | Round-1 sequence design (ProteinMPNN / AbMPNN) |
| `flowpacker1` | Side-chain packing (FlowPacker) |
| `af3score1` | Round-1 scoring (AlphaFold 3 / AF3Score) |
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

</details>

For step-by-step details (inputs/outputs and per-step configs), see `documentation/pipeline_steps.md`.

## Install

Clone this repo:
```bash
git clone https://github.com/cytokineking/PPIFlow-Pipeline
cd PPIFlow-Pipeline
```

Run the installer:
```bash
./install_ppiflow.sh \
  --af3-weights-path /path/to/af3.bin.zst \
  --ppiflow-checkpoints-path /path/to/ppiflow_checkpoints
```

Activate:
```bash
source ./env.sh
conda activate ppiflow
```

Installer notes:
- No conda? Add `--install-conda` (installs Miniforge).
- Fresh Ubuntu/Debian node? Add `--install-os-deps` (runs `apt-get`).
- To skip writing `env.sh`: `--no-write-env`.

## Quick Test (Bundled PDL1 Examples)

We bundle a small target PDB (`examples/targets/PDL1.pdb`) plus runnable example YAMLs so you can validate installation + tool wiring end-to-end.

Binder:
```bash
python ppiflow.py pipeline --input examples/pdl1_binder.yaml --output runs/example_pdl1_binder
```

VHH:
```bash
python ppiflow.py pipeline --input examples/pdl1_vhh.yaml --output runs/example_pdl1_vhh
```

Antibody (scFv):
```bash
python ppiflow.py pipeline --input examples/pdl1_antibody.yaml --output runs/example_pdl1_antibody
```

These examples are intentionally small (`samples_per_target: 10`). They are for validating that the pipeline runs, not for producing high-quality designs.

See `examples/README.md` for details.

## Key Concepts (The Minimum You Need)

- **Protocols:** `binder` designs a free-form binder chain; `vhh` designs a nanobody (single heavy chain); `antibody` designs an scFv (heavy + light).
- **Hotspots syntax:** chain is required. Examples: `A` (whole chain), `A56` (single residue), `A56-70` (range), `A3,A5-25,B72` (mixed list). Hotspots use chain IDs/residue numbers from your input PDB; they are mapped to the internal concatenated target chain `B` during `configure`.
- **Chain conventions (internal):**
  - Binder/VHH: binder is `A`, target is `B`
  - Antibody: heavy is `A`, light is `C`, target is `B`
- **Framework chain IDs:** `framework.heavy_chain` / `framework.light_chain` should match the chain IDs in the framework PDB you provide (many scFv frameworks are `A/B`). During `configure`, PPIFlow rewrites framework chains to the internal conventions (heavy `A`, light `C`).
- **Multi-chain targets:** if you provide multiple target chains, the pipeline concatenates them into a single gapped internal target chain `B` during `configure`. Hotspots are specified on the original target chain IDs/residue numbers and then mapped onto the concatenated internal chain `B`.
- **`samples_per_target`:** number of backbones to generate. The PPIFlow authors recommend ~20k+ for serious production runs. This is where multi-GPU parallelization matters most.

## Run Your Own Job

### YAML (Recommended)

Start from one of:
- `assets/examples/binder_minimal.yaml`
- `assets/examples/vhh_minimal.yaml`
- `assets/examples/antibody_minimal.yaml`

Run:
```bash
python ppiflow.py pipeline --input /path/to/design.yaml --output /path/to/out_dir
```

### CLI-Only (No YAML) Binder Example

```bash
python ppiflow.py pipeline \
  --protocol binder \
  --name my_first_binder \
  --target_pdb /path/to/target.pdb \
  --target_chains A \
  --binder_length 75-90 \
  --hotspots A56 \
  --output /path/to/out_dir
```

For antibody/VHH CLI-only required flags, see `documentation/cli_reference.md`.

## Parallelize (GPU + CPU)

Multi-GPU:
```bash
python ppiflow.py pipeline --input design.yaml --output out_dir --num-devices all
```

Pin to specific GPUs:
```bash
python ppiflow.py pipeline --input design.yaml --output out_dir --devices 0,2,3 --num-devices 3
```

Rosetta CPU workers (CPU-heavy steps):
```bash
python ppiflow.py execute --output out_dir --num-rosetta-workers 32
```

Guideline: set `--num-rosetta-workers` to roughly your available CPU cores; a practical upper bound is around ~40 on many systems.

More details: `documentation/scaling_and_orchestration.md`.

## Resume

Resume an interrupted run:
```bash
python ppiflow.py execute --output /path/to/out_dir
```

More recovery patterns: `documentation/resume_and_recovery.md`.

## Where Are My Results?

- Ranked results: `results/`
- Step outputs / intermediates: `output/`
- Logs: `logs/`

Full layout reference: `documentation/output_layout.md`.

## Antibody & VHH Frameworks

Bundled frameworks live in `assets/frameworks/`. For a copy/paste inventory (VHH + scFv) and the exact `--framework_pdb/--cdr_length` blocks, see `documentation/frameworks.md`.

## Troubleshooting (Short List)

- `steps.yaml not found`: run `pipeline` (or `configure`) first.
- `pipeline_state.json ... mismatch`: use a new output directory or ensure tool paths/inputs match the original run.
- Lock errors: use `--steal-lock` if the old job is dead.

## Documentation Index

- `documentation/cli_reference.md`
- `documentation/frameworks.md`
- `documentation/pipeline_steps.md`
- `documentation/scaling_and_orchestration.md`
- `documentation/output_layout.md`
- `documentation/resume_and_recovery.md`
- `examples/README.md`

## License

The original PPIFlow code is licensed under an Attribution-NonCommercial-ShareAlike 4.0 International license. This pipeline is licensed under the same license to respect the upstream repository.
