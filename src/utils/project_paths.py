from __future__ import annotations

import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"
PROJECT_TMP_DIR = LOGS_DIR / "tmp"
PROJECT_CACHE_DIR = LOGS_DIR / "cache"


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def project_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not is_relative_to(candidate, PROJECT_ROOT):
        raise ValueError(f"Refusing to use path outside project root: {candidate}")
    return candidate


def output_path(path: str | Path = "") -> Path:
    return project_path(path, OUTPUTS_DIR)


def experiment_path(path: str | Path = "") -> Path:
    return project_path(path, EXPERIMENTS_DIR)


def log_path(path: str | Path = "") -> Path:
    return project_path(path, LOGS_DIR)


def artifact_path(path: str | Path, default_base: Path = OUTPUTS_DIR) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    elif candidate.parts and candidate.parts[0] in {"outputs", "experiments", "logs"}:
        resolved = project_path(candidate)
    else:
        resolved = project_path(candidate, default_base)

    allowed_roots = [OUTPUTS_DIR, EXPERIMENTS_DIR, LOGS_DIR]
    if not any(is_relative_to(resolved, allowed) for allowed in allowed_roots):
        raise ValueError(
            "Project artifacts must be written under outputs, experiments, or logs: "
            f"{resolved}"
        )
    return resolved


def ensure_project_storage() -> dict[str, Path]:
    directories = {
        "project_root": PROJECT_ROOT,
        "outputs": OUTPUTS_DIR,
        "experiments": EXPERIMENTS_DIR,
        "logs": LOGS_DIR,
        "tmp": PROJECT_TMP_DIR,
        "cache": PROJECT_CACHE_DIR,
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(PROJECT_CACHE_DIR / "matplotlib")
    os.environ["TORCH_HOME"] = str(PROJECT_CACHE_DIR / "torch")
    os.environ["XDG_CACHE_HOME"] = str(PROJECT_CACHE_DIR)
    os.environ["HF_HOME"] = str(PROJECT_CACHE_DIR / "huggingface")
    os.environ["WANDB_DIR"] = str(LOGS_DIR / "wandb")
    os.environ["WANDB_CACHE_DIR"] = str(PROJECT_CACHE_DIR / "wandb")
    os.environ["WANDB_CONFIG_DIR"] = str(PROJECT_CACHE_DIR / "wandb_config")
    os.environ["MNE_DATA"] = str(PROJECT_CACHE_DIR / "mne")
    os.environ["TMP"] = str(PROJECT_TMP_DIR)
    os.environ["TEMP"] = str(PROJECT_TMP_DIR)
    os.environ["TMPDIR"] = str(PROJECT_TMP_DIR)
    tempfile.tempdir = str(PROJECT_TMP_DIR)

    for env_path in [
        Path(os.environ["MPLCONFIGDIR"]),
        Path(os.environ["TORCH_HOME"]),
        Path(os.environ["XDG_CACHE_HOME"]),
        Path(os.environ["HF_HOME"]),
        Path(os.environ["WANDB_DIR"]),
        Path(os.environ["WANDB_CACHE_DIR"]),
        Path(os.environ["WANDB_CONFIG_DIR"]),
        Path(os.environ["MNE_DATA"]),
    ]:
        env_path.mkdir(parents=True, exist_ok=True)

    return directories


def print_storage_summary(extra: dict[str, str | Path] | None = None) -> None:
    directories = ensure_project_storage()
    print("Storage summary", flush=True)
    for name, path in directories.items():
        print(f"  {name}: {path}", flush=True)
    if extra:
        for name, path in extra.items():
            resolved = project_path(path)
            print(f"  {name}: {resolved}", flush=True)
