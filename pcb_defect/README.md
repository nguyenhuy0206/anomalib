# DeepPCB Experiments

This folder contains reusable Anomalib experiments for PCB defect detection on the
DeepPCB dataset.

## Layout

```text
pcb_defect/
  core/
    config.py      # dataclass configs and experiment presets
    data.py        # Anomalib Folder datamodule builder
    models.py      # model builders: patchcore, patchcore_simam, padim
    prepare.py     # DeepPCB -> Folder layout converter
    runner.py      # shared train/test/checkpoint logic
  run_experiment.py
  train_patchcore_simam_pcb.py
  test_patchcore_simam_dpcb.py
  train_padim_pcb.py
  test_padim_pcb.py
```

The old script names are kept as wrappers, so existing commands still work. New
experiments should go through `core/config.py` and `core/models.py`.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m pcb_defect.run_experiment prepare
python -m pcb_defect.run_experiment train-test --experiment patchcore_simam
```

Equivalent wrapper:

```bash
python -m pcb_defect.train_patchcore_simam_pcb
```

Test the latest checkpoint:

```bash
python -m pcb_defect.run_experiment test --experiment patchcore_simam
```

PaDiM baseline:

```bash
python -m pcb_defect.run_experiment train --experiment padim
python -m pcb_defect.run_experiment test --experiment padim
```

Defaults use `num_workers=0` because it is more reliable on macOS and sandboxed
Python environments. If your local machine supports multiprocessing, override it.

If memory is tight, lower the batch size:

```bash
python -m pcb_defect.run_experiment train-test --experiment patchcore_simam --train-batch-size 4 --num-workers 4
```

## Adding More Anomalib Models

1. Add a preset in `core/config.py`.
2. If the model needs custom setup, add a builder in `core/models.py` and register
   it in `MODEL_BUILDERS`.
3. If it can be constructed directly by Anomalib, set `ModelConfig.name` to the
   Anomalib model name and pass model-specific args through `init_args`.

## Dataset Structure

The converter creates:

```text
pcb_defect/data/
  train/normal/
  test/normal/
  test/abnormal/
```

`train/normal` contains DeepPCB template images for fitting one-class models.
`test/abnormal` contains DeepPCB defect images for evaluation.
`test/normal` contains template images from the held-out test split.
