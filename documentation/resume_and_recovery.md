# Resume and Recovery

## The Mental Model

- A run is an `--output` directory.
- To resume the same run, rerun `execute` (or `pipeline`) pointing at the same `--output`.
- The pipeline skips completed work and continues from the first incomplete step.

## Common Commands

Resume:
```bash
python ppiflow.py execute --output /path/to/out_dir
```

Resume with multiple GPUs:
```bash
python ppiflow.py execute --output /path/to/out_dir --num-devices all
```

Retry failed items in work-queue steps:
```bash
python ppiflow.py execute --output /path/to/out_dir --retry-failed
```

Rebuild the work queue DB from outputs on disk (use if `.work/` is missing/corrupt or progress looks wedged):
```bash
python ppiflow.py execute --output /path/to/out_dir --work-queue-rebuild
```

## Output Locking

PPIFlow uses a run lock under `.ppiflow_lock/` to prevent two controllers from writing to the same output directory simultaneously.

If you are sure the previous controller is dead:
```bash
python ppiflow.py execute --output /path/to/out_dir --steal-lock
```

## Identity Mismatches

`pipeline_state.json` enforces run identity. If you change inputs/tool paths in ways that change the run hash/stamps, you may need a new output directory rather than resuming.

## Deep Dive

See:
- `documentation/scaling_and_orchestration.md` for multi-GPU + worker pool behavior.
- `documentation/output_layout.md` for where the state files, locks, and work-queue DB live.

