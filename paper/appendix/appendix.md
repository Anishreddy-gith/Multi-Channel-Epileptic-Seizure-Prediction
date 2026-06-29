# Appendix

## A. Artifact Index

The full master index is available in:

- `paper/master_experiment_index.json`
- `paper/tables/master_experiment_index.csv`
- `paper/tables/master_experiment_index.md`
- `paper/tables/master_experiment_index.tex`

A human-readable summary is available in `paper/MASTER_EXPERIMENT_INVENTORY.md`.

## B. Additional Tables

The paper package includes CSV, Markdown, and LaTeX versions of:

- Dataset summary
- Patient split
- Class distribution by patient
- Model configuration
- Training configuration
- Hardware and software configuration
- Model comparison
- Optimization summary
- Label audit summary
- Preictal horizon dataset statistics
- Preictal horizon ablation results

## C. Additional Figures

The package includes PNG, PDF, and SVG versions where available. Figure dimensions and usage notes are documented in `paper/figures/FIGURE_AUDIT.md`.

## D. Reproducibility Notes

The completed experiments used fixed seed 42, AdamW, mixed precision, ReduceLROnPlateau scheduling, early stopping, and checkpointing. CUDA bitwise determinism is not guaranteed because CuBLAS deterministic settings were not available for all completed runs.

## E. Unavailable Evidence

The frozen repository does not contain leave-one-patient-out validation, repeated-seed variance, confidence intervals, seizure-level false-alarm rates, or classical feature-based baseline results. These are listed as limitations rather than inferred.
