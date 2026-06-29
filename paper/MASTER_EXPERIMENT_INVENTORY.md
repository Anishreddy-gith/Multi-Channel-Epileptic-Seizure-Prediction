# Master Experiment Inventory

Generated for the final paper package from existing repository artifacts only. No source code, preprocessing output, labels, splits, checkpoints, metrics, or experiment results were modified.

## Indexed Locations

- `paper/`: manuscript, LaTeX, tables, figures, captions, references, supplementary material, appendix, and review files.
- `outputs/`: smoke tests, data audit outputs, label audit outputs, class-imbalance study outputs, optimization-cycle outputs, preictal-horizon ablation outputs, and copied paper figures.
- `experiments/`: checkpoints, run configurations, histories, split metadata, TensorBoard event files, and horizon-ablation label shards.
- `logs/`: progress log, decision log, local cache metadata, and overnight run logs.

## Artifact Counts

| Area | Extension | Count |
| --- | ---: | ---: |
| experiments | `.npy` | 2184 |
| experiments | `.csv` | 1118 |
| experiments | `.pth` | 35 |
| experiments | `.json` | 17 |
| experiments | `.pkl` | 2 |
| outputs | `.png` | 167 |
| outputs | `.csv` | 70 |
| outputs | `.json` | 32 |
| outputs | `.npz` | 14 |
| outputs | `.md` | 10 |
| paper | `.md` | 25 |
| paper | `.png` | 17 |
| paper | `.tex` | 17 |
| paper | `.csv` | 15 |
| paper | `.pdf` | 12 |
| paper | `.svg` | 12 |
| logs | `.md` | 2 |
| logs | `.log` | 2 |

The full row-level experiment index is stored in:

- `paper/master_experiment_index.json`
- `paper/tables/master_experiment_index.csv`
- `paper/tables/master_experiment_index.md`
- `paper/tables/master_experiment_index.tex`

## Key Experiment Sources

| Evidence type | Source artifact |
| --- | --- |
| Dataset audit | `outputs/paper/data_audit_report.csv`, `outputs/paper/data_audit_report.json` |
| Label audit | `outputs/paper/label_audit/LABEL_AUDIT_REPORT.md`, `outputs/paper/label_audit/label_audit_summary.json` |
| Imbalance study | `outputs/paper/overnight_20260628_034710/metrics/`, `outputs/paper/overnight_20260628_034710/status/` |
| Final model comparison | `outputs/paper/overnight_20260628_034710/tables/final_metrics.csv`, `paper/tables/model_comparison_table.csv` |
| Optimization cycle | `outputs/paper/optimization_cycle_20260628_191139/`, `paper/tables/optimization_summary_table.csv` |
| Horizon ablation | `outputs/paper/preictal_horizon_ablation/PREICTAL_HORIZON_ABLATION_REPORT.md`, `paper/tables/preictal_horizon_ablation_table.csv` |
| Reproducibility metadata | `paper/supplementary/supplementary_material.md`, `logs/decision_log.md`, `logs/progress.md` |

## Consistency Check

The model-comparison table was checked against `outputs/paper/overnight_20260628_034710/tables/final_metrics.csv` for overlapping test metrics and thresholds.

Result: **0 inconsistencies found**.

The paper tables preserve measured values from the stored CSV/JSON artifacts. Any unavailable information, including confidence intervals and repeated-seed variance, is explicitly marked as unavailable in the manuscript and review files rather than inferred.
