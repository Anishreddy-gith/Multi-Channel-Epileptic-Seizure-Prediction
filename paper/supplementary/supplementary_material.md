# Supplementary Material

## Hyperparameters

| seed | epochs | batch_size | num_workers | lr | weight_decay | focal_gamma | optimizer | scheduler | amp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 42 | 25 | 128 | 0 | 0.0003 | 0.0001 | 2.0 | AdamW | ReduceLROnPlateau | True |

## Random Seeds

Main seed: `42`. CUDA deterministic warnings were observed for CuBLAS operations, so bitwise reproducibility may require setting `CUBLAS_WORKSPACE_CONFIG` before process start. The paper therefore claims repository-backed reproducibility artifacts, not guaranteed bitwise reproduction.

## Hardware

| component | value |
| --- | --- |
| OS | Windows-10-10.0.26200-SP0 |
| Python | 3.11.9 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| CUDA device count | 1 |
| nvidia-smi | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 581.95 |

## Software

| package | version |
| --- | --- |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| torch | 2.3.1+cu118 |
| sklearn | 1.5.1 |
| mne | 1.12.1 |
| matplotlib | 3.9.2 |
| scipy | 1.16.1 |

## Directory Structure

- `data/interim/windows/`: frozen per-EDF shards.
- `data/processed/splits/`: frozen patient-wise split metadata.
- `outputs/paper/overnight_20260628_034710/`: main experiment.
- `outputs/paper/label_audit/`: label audit.
- `outputs/paper/preictal_horizon_ablation/`: horizon ablation.
- `paper/`: generated submission package.

## Reproducibility

No new training or experiments were run while generating the final paper package. Metrics were loaded from saved CSV and JSON artifacts. Missing quantities, including confidence intervals, repeated-seed variance, and seizure-level false-alarm rates, are marked unavailable rather than estimated.

## Suggested Reproduction Commands

The exact completed commands are not fully reconstructed in the paper package. The available entry points and artifacts indicate the following repository-level workflow:

```bash
python src/training/overnight_experiment_runner.py
python src/analysis/label_audit.py
python src/training/preictal_horizon_ablation.py
```

If these scripts are rerun, results may differ because CUDA bitwise determinism was not guaranteed in the completed artifacts.
