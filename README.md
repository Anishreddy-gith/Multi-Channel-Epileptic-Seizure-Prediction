# Explainable Hybrid EEGNet-Transformer Architecture for Multi-Channel Epileptic Seizure Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

A lightweight, explainable deep learning framework for epileptic seizure prediction using multi-channel EEG from the CHB-MIT Scalp EEG Database. The pipeline combines **EEGNet** (spatial feature extraction), a **Transformer encoder** (temporal modeling), and **SHAP** (channel-wise interpretability) to classify EEG windows as **preictal** vs **interictal**.

Submitted to **IEEE EMBS Pune Chapter**.

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Preprocessing Protocol](#preprocessing-protocol)
- [Recommended Repository Structure](#recommended-repository-structure)
- [Installation](#installation)
- [Dataset Download (CHB-MIT)](#dataset-download-chb-mit)
- [Usage](#usage)
- [Evaluation Protocol](#evaluation-protocol)
- [Results (Placeholder)](#results-placeholder)
- [Explainability with SHAP](#explainability-with-shap)
- [Citation](#citation)
- [Team](#team)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Overview
- **Task:** Binary EEG segment classification (`preictal`, `interictal`)
- **EEG Source:** CHB-MIT Scalp EEG Database (PhysioNet)
- **Patients:** `chb01` to `chb05`
- **Sampling Rate:** 256 Hz
- **Segment Size:** 4 s (1024 samples)
- **Stride:** 0.5 s
- **Labels:**
  - **Preictal:** 0–30 min before seizure onset
  - **Interictal:** >240 min away from any seizure onset
- **Split Strategy:** 70/15/15 at **recording-file level** (not window level)

## System Architecture

```text
CHB-MIT .edf + seizure summaries
            |
            v
   Channel Selection (18 common EEG channels)
            |
            v
   Bandpass Filter (0.5–40 Hz, FIR)
            |
            v
      Notch Filter (50 Hz)
            |
            v
Sliding Window Segmentation (4 s, 0.5 s stride)
            |
            v
Z-score Normalization (per channel, per segment)
            |
            v
        EEGNet Backbone
   (spatial/channel feature extraction)
            |
            v
     Transformer Encoder
 (temporal dependency modeling)
            |
            v
  Dense + Softmax Classifier
 (preictal vs interictal output)
            |
            v
 SHAP DeepExplainer Analysis
(channel-wise clinical interpretability)
```

## Preprocessing Protocol
- Select 18 common channels across subjects
- FIR bandpass filtering: 0.5–40 Hz
- Notch filtering: 50 Hz
- Sliding windows: 4 s, stride 0.5 s
- Channel-wise z-score normalization per segment
- File-level data split to prevent leakage

## Recommended Repository Structure

```text
Multi-Channel-Epileptic-Seizure-Prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                    # CHB-MIT EDF files (ignored)
│   ├── interim/                # Intermediate preprocessing outputs (ignored)
│   └── processed/              # Model-ready tensors/features (ignored)
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_shap_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── download_chbmit.py
│   │   ├── parse_annotations.py
│   │   └── preprocess_eeg.py
│   ├── models/
│   │   ├── eegnet.py
│   │   ├── transformer_encoder.py
│   │   └── hybrid_eegnet_transformer.py
│   ├── training/
│   │   ├── train.py
│   │   └── losses_metrics.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── cross_patient_validation.py
│   └── explainability/
│       ├── shap_explainer.py
│       └── plot_shap_channels.py
├── experiments/
│   ├── configs/
│   ├── logs/                   # ignored
│   └── checkpoints/            # ignored
└── outputs/
    ├── figures/
    └── reports/
```

## Installation

```bash
git clone https://github.com/Anishreddy-gith/Multi-Channel-Epileptic-Seizure-Prediction.git
cd Multi-Channel-Epileptic-Seizure-Prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Google Colab:
```bash
!pip install -r requirements.txt
```

## Dataset Download (CHB-MIT)
1. Go to PhysioNet: https://physionet.org/content/chbmit/1.0.0/
2. Request/sign in if prompted by PhysioNet.
3. Download subject folders `chb01` to `chb05`.
4. Place downloaded `.edf` and summary annotation files under `data/raw/`.
5. Keep raw files unchanged; store derived outputs in `data/interim/` or `data/processed/`.

> **License note:** CHB-MIT data is distributed under **ODC-By v1.0**.

## Usage

### 1) Preprocessing
```bash
python src/data/preprocess_eeg.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --patients chb01 chb02 chb03 chb04 chb05 \
  --window_sec 4 --stride_sec 0.5 \
  --bandpass_low 0.5 --bandpass_high 40 --notch 50
```

### 2) Training
```bash
python src/training/train.py \
  --data_dir data/processed \
  --model hybrid_eegnet_transformer \
  --epochs 100 --batch_size 64 --learning_rate 1e-3
```

### 3) Evaluation
```bash
python src/evaluation/evaluate.py \
  --data_dir data/processed \
  --checkpoint experiments/checkpoints/best_model.h5 \
  --metrics accuracy precision recall f1 roc_auc fpr_per_hour
```

### 4) SHAP Explainability
```bash
python src/explainability/shap_explainer.py \
  --checkpoint experiments/checkpoints/best_model.h5 \
  --data_dir data/processed \
  --output_dir outputs/figures/shap
```

## Evaluation Protocol
- Within-patient validation
- Cross-patient validation (leave-one-patient-out)
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - False Positive Rate/hour

## Results (Placeholder)

| Setting | Accuracy | Precision | Recall | F1-Score | ROC-AUC | FPR/hour |
|--------|----------|-----------|--------|----------|---------|----------|
| Within-patient (chb01–chb05) | TBD | TBD | TBD | TBD | TBD | TBD |
| Cross-patient (LOPO) | TBD | TBD | TBD | TBD | TBD | TBD |

## Explainability with SHAP
- SHAP `DeepExplainer` is used to estimate feature contribution per channel.
- Output artifacts:
  - Channel importance bar plots
  - Per-patient SHAP summary plots
- Suggested storage:
  - `outputs/figures/shap/channel_importance_*.png`
  - `outputs/figures/shap/summary_*.png`

## Citation

```bibtex
@misc{nelabhotla2026explainable,
  title={Explainable Hybrid EEGNet-Transformer Architecture for Multi-Channel Epileptic Seizure Prediction},
  author={Nelabhotla, Alekhya and Reddy, Boppidi Anish and Kademgari, Avanika},
  year={2026},
  note={Submitted to IEEE EMBS Pune Chapter}
}
```

## Team
- Alekhya Nelabhotla
- Boppidi Anish Reddy
- Avanika Kademgari

## Acknowledgements
- IEEE EMBS Pune Chapter
- PhysioNet CHB-MIT Scalp EEG Database contributors
- Open-source community: TensorFlow/Keras, MNE-Python, SHAP, SciPy, NumPy, Matplotlib, Seaborn

## License
This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
