# Master Experiment Inventory

This inventory summarizes the repository artifacts that are intentionally kept in Git for the final publication package. Raw data, generated tensors, checkpoints, experiment directories, logs, and local outputs are intentionally ignored.

## Final Paper Package

- `paper/conference_paper.pdf`
- `paper/conference_paper.md`
- `paper/latex/main.tex`
- `paper/latex/abstract.tex`
- `paper/latex/build_pdf.ps1`
- `paper/references/references.bib`
- `paper/figures/`
- `paper/tables/`
- `paper/supplementary/`
- `paper/appendix/`
- `paper/review/`

## Demo Notebook

- `notebooks/00_complete_project_demo.ipynb`
- `notebooks/README.md`
- `notebooks/requirements_demo.txt`

The four earlier split demo notebooks were consolidated into the single complete demo notebook to avoid duplication and make the repository easier to present.

## Generated Inventories

- `paper/master_experiment_index.json`
- `paper/tables/master_experiment_index.csv`
- `paper/tables/master_experiment_index.md`
- `paper/tables/master_experiment_index.tex`
- `paper/tables/paper_package_manifest.csv`
- `paper/tables/paper_package_manifest.md`
- `paper/tables/paper_package_manifest.tex`

These files were regenerated from the current repository state during the final cleanup pass.

## Notes

- The final conference PDF was built from `paper/latex/main.tex`.
- LaTeX build intermediates are ignored and not tracked.
- `paper/latex/main.pdf` is treated as a local build artifact; the tracked final PDF is `paper/conference_paper.pdf`.
- Large runtime artifacts remain local-only by design.
