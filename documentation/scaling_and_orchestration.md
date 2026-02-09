# Scaling and Orchestration

## GPU Parallelism (`--num-devices`)

Use `--num-devices` to parallelize GPU-heavy steps across devices (one worker per GPU):

```bash
python ppiflow.py pipeline --input design.yaml --output out_dir --num-devices all
```

Pin to specific GPUs:
```bash
python ppiflow.py pipeline --input design.yaml --output out_dir --devices 0,2,3
```

`--devices all` is equivalent to `--num-devices all`.

Note: `--devices 0,2,3` implies `--num-devices 3` (you can omit `--num-devices`).

## Rosetta CPU Workers (`--num-rosetta-workers`)

Rosetta-heavy item steps can be accelerated by increasing the CPU worker pool size:

```bash
python ppiflow.py execute --output out_dir --num-rosetta-workers 32
```

Guideline: start near your available CPU cores; a practical upper bound is around ~40 on many systems.

Alias: `--num-cpu-workers`.

## Orchestrate Subcommand

You typically do not need to call `orchestrate` directly. `pipeline` and `execute` use the orchestrator controller by default.

To force the legacy single-process execution path (debug/regression only), pass `--single-process` to `pipeline` or `execute`.

The orchestrator writes artifacts under `.orchestrator/` in your output directory.

## Precedence (CLI vs YAML)

If both are provided, CLI flags win over YAML defaults:

- `--devices ...` (explicit pinning) overrides everything else
- `--num-devices N` / `--num-devices all` overrides YAML `orchestrator.gpu_binding.devices`
- YAML `orchestrator.gpu_binding.devices` is used only when you do not pass GPU-selection flags
