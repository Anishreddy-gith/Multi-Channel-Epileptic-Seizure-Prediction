from __future__ import annotations

import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = LOGS_DIR / "cache"
TMP_DIR = LOGS_DIR / "tmp"

for path in [
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "experiments",
    LOGS_DIR,
    CACHE_DIR,
    TMP_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["WANDB_DIR"] = str(LOGS_DIR / "wandb")
os.environ["WANDB_CACHE_DIR"] = str(CACHE_DIR / "wandb")
os.environ["WANDB_CONFIG_DIR"] = str(CACHE_DIR / "wandb_config")
os.environ["MNE_DATA"] = str(CACHE_DIR / "mne")
tempfile.tempdir = str(TMP_DIR)

for env_name in [
    "MPLCONFIGDIR",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_CONFIG_DIR",
    "MNE_DATA",
]:
    Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)
