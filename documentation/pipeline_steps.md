# Pipeline Steps

This page is the reference for what each step does, where outputs go, and which steps require external tools.

## Step Overview

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

## Skipping Refold (Stopgap)

When running without a refold policy/installation, you can use `--skip-refold` (aka `--skip-af3-refold`) to:

- omit `af3_refold` and `dockq` from the configured/executed step set
- force ranking to use AF3Score R2 metrics + AF3Score R2 predicted structures (so scores and exported structures are consistent)

## Step Outputs

Most step outputs live under `output/`. Ranking outputs live under `results/`.

Some step names don't match their output directory:

| Step name | Output directory |
|---|---|
| `partial` | `output/partial_flow` |
| `rank_features` | `results/features` |
| `rank_finalize` | `results/` |
| `rosetta_interface2` | `output/rosetta_interface2` |

## External Tool Wiring

The pipeline orchestrates multiple external tools. Tool paths can be discovered via:

1. Your YAML `tools:` block (highest priority).
2. Environment variables exported by `env.sh` (recommended).
3. Repo-local defaults under `assets/` created by `./install_ppiflow.sh`.

After `configure`, per-step configs are written under `config/`. Most users should not need to edit these, but they are the escape hatch for clusters/custom installs.

Key step config files:
- `config/step_seq1.yaml`
- `config/step_flowpacker1.yaml`
- `config/step_af3score1.yaml`
- `config/step_rosetta_interface.yaml`
- `config/step_interface_enrich.yaml`
- `config/step_seq2.yaml`
- `config/step_flowpacker2.yaml`
- `config/step_af3score2.yaml`
- `config/step_relax.yaml`
- `config/step_rosetta_interface2.yaml`
- `config/step_af3_refold.yaml`
- `config/step_dockq.yaml`

Each step config may contain a `command` field (string or list) executed as a subprocess during `execute`.

## Multi-Chain Targets and Hotspots

- You provide `target.chains` and optional `target.hotspots` using the chain IDs and residue numbering of your original input PDB.
- During `configure`, if multiple target chains are provided, they are concatenated into a single gapped internal target chain `B`.
- Hotspots are expanded on the original target PDB, then mapped into the internal concatenated chain `B`. Mapping artifacts are written under `inputs/`.
