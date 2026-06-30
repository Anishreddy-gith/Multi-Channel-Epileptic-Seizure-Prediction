# Multi-Channel Epileptic Seizure Prediction

Patient-wise CHB-MIT seizure prediction study using multi-channel scalp EEG, sharded preprocessing, PyTorch training, leakage audits, label audits, imbalance experiments, preictal horizon ablation, and a final conference-paper package.

The scientific outputs are frozen. The repository keeps source code, documentation, demo notebooks, and publication assets. Raw data, generated tensors, checkpoints, experiment folders, and runtime logs are intentionally excluded from Git.

## Project Summary

- Dataset: CHB-MIT Scalp EEG Database subset
- Patients: 10
- EDF recordings: 273
- Seizure events: 50
- Windows: 734,796 four-second windows
- Channels: 18 common EEG channels
- Windowing: 4 s windows with 2 s stride
- Filtering: 0.5--40 Hz bandpass and 50 Hz notch
- Normalization: per-window, per-channel z-score normalization
- Primary label rule: 10 min preictal horizon with 60 s postictal exclusion
- Split: patient-wise train/validation/test split
- Framework: PyTorch

## Main Findings

The completed experiments did not support strong window-level patient-wise seizure prediction under the frozen protocol.

| Model | Status | Validation PR-AUC | Validation F1 | Test PR-AUC | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| EEGNet | Completed | 0.0239 | 0.0366 | 0.0179 | 0.0258 |
| Attention | Completed | 0.0258 | 0.0000 | 0.0163 | 0.0000 |
| Hybrid EEGNet-Transformer | Incomplete | 0.0238 | 0.0461 | 0.0181 | 0.0354 |

The positive class prevalence under the 10 min preictal definition was 1.9250%. The label audit found no evidence of mechanical label contamination under the implemented checks, but 47.26% of positive windows occurred more than five minutes before seizure onset. A controlled preictal horizon ablation over 10, 7.5, 5, and 3 minutes did not find a shorter horizon that improved both validation PR-AUC and validation F1.

These results are reported as a bounded negative finding, not as a clinical deployment claim or state-of-the-art result.

## Repository Structure

```text
.
|-- data/                    # Local raw/interim/processed data; ignored by Git
|-- experiments/             # Local checkpoints and run artifacts; ignored by Git
|-- logs/                    # Local run logs and cache metadata; ignored by Git
|-- notebooks/               # Single combined demo notebook and setup guide
|-- outputs/                 # Local generated outputs; ignored by Git
|-- paper/                   # Conference paper package and final PDF
|-- src/
|   |-- data/                # EDF parsing and preprocessing scripts
|   |-- evaluation/          # Evaluation and threshold scripts
|   |-- explainability/      # SHAP-related scripts
|   |-- models/              # EEGNet, Attention, and Hybrid models
|   |-- training/            # Training, losses, sharded dataset, split utilities
|   `-- utils/               # Project path utilities
`-- tools/                   # Helper scripts
```

## Paper Package

Key publication files:

- `paper/conference_paper.pdf`
- `paper/latex/main.tex`
- `paper/latex/abstract.tex`
- `paper/latex/build_pdf.ps1`
- `paper/references/references.bib`
- `paper/figures/`
- `paper/tables/`
- `paper/supplementary/`
- `paper/appendix/`
- `paper/review/`

To rebuild the PDF on Windows after installing MiKTeX:

```powershell
powershell -ExecutionPolicy Bypass -File paper\latex\build_pdf.ps1
```

## Demo Notebook

For a video walkthrough, use:

```text
notebooks/00_complete_project_demo.ipynb
```

The notebook combines preprocessing, training, evaluation, label audit, preictal horizon ablation, and explainability/package review into one safe demo. It displays existing validated artifacts and does not rerun full preprocessing or training.

Notebook setup instructions are in `notebooks/README.md`. A lightweight demo dependency list is available at `notebooks/requirements_demo.txt`.

## Installation

```powershell
git clone https://github.com/Anishreddy-gith/Multi-Channel-Epileptic-Seizure-Prediction.git
cd Multi-Channel-Epileptic-Seizure-Prediction
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Access

CHB-MIT is available from PhysioNet:

https://physionet.org/content/chbmit/1.0.0/

Raw EDF files should be placed under `data/raw/` locally. Raw data and generated artifacts are intentionally ignored by Git.

## Reproducibility Notes

This repository contains code and paper artifacts, but raw CHB-MIT data and generated tensors/checkpoints are not tracked in Git. CUDA bitwise determinism was not guaranteed in the completed runs. Missing evidence, including confidence intervals, leave-one-patient-out validation, and seizure-level false-alarm rates, is documented as unavailable rather than estimated.

## License

This repository is licensed under the MIT License. See `LICENSE` for details.
