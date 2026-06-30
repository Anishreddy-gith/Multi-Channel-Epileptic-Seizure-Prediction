# Demo Notebooks

Use `00_complete_project_demo.ipynb` for the demo video. It combines the preprocessing, training, evaluation, and explainability/package review notebooks into one safe walkthrough.

The notebook displays existing validated results and instantiates the three model families. It does not rerun preprocessing, full training, or evaluation.

## Recommended Setup

From the repository root:

```powershell
cd D:\epilepsy\seizure-prediction
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name seizure-prediction --display-name "Seizure Prediction (.venv)"
jupyter lab
```

Then open:

```text
notebooks/00_complete_project_demo.ipynb
```

If you only need the demo notebook dependencies in a fresh environment, install:

```powershell
python -m pip install -r notebooks\requirements_demo.txt
```

For GPU execution, keep the existing project environment if possible. If PyTorch installation fails in a new environment, install the PyTorch build appropriate for your CUDA version, then rerun the remaining requirements.
