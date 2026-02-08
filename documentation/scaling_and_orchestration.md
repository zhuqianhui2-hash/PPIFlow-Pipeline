# Scaling and Orchestration

## GPU Parallelism (`--num-devices`)

Use `--num-devices` to parallelize GPU-heavy steps across devices (one worker per GPU):

```bash
python ppiflow.py pipeline --input design.yaml --output out_dir --num-devices all
```

Pin to specific GPUs:
```bash
python ppiflow.py pipeline --input design.yaml --output out_dir --devices 0,2,3 --num-devices 3
```

## Rosetta CPU Workers (`--num-rosetta-workers`)

Rosetta-heavy item steps can be accelerated by increasing the CPU worker pool size:

```bash
python ppiflow.py execute --output out_dir --num-rosetta-workers 32
```

Guideline: start near your available CPU cores; a practical upper bound is around ~40 on many systems.

Alias: `--num-cpu-workers`.

## Orchestrate Subcommand

You typically do not need to call `orchestrate` directly. When you pass `--num-devices` and/or `--num-rosetta-workers` to `pipeline` or `execute`, the CLI will route internally to the orchestrator.

The orchestrator writes artifacts under `.orchestrator/` in your output directory.

