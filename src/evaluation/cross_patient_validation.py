from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.sharded_dataset import discover_shards, load_shard_metadata
from src.training.train import DEFAULT_CONFIG, load_config, train_model
from src.utils.project_paths import artifact_path, ensure_project_storage, experiment_path, print_storage_summary

OUTPUT_COLUMNS = [
    "dataset_index",
    "shard_id",
    "shard_row",
    "patient_id",
    "label",
    "file_name",
]


def patient_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metadata.groupby("patient_id")["label"]
        .agg(total="size", positive="sum")
        .reset_index()
    )
    summary["positive_ratio"] = summary["positive"] / summary["total"]
    return summary


def make_patient_folds(metadata: pd.DataFrame, max_folds: int | None = None):
    patients = sorted(metadata["patient_id"].unique())
    if max_folds is not None:
        patients = patients[:max_folds]

    for test_patient in patients:
        remaining = [patient for patient in sorted(metadata["patient_id"].unique()) if patient != test_patient]
        if len(remaining) < 2:
            raise ValueError("Need at least 3 patients for patient-wise CV")

        train_pool = metadata[metadata["patient_id"].isin(remaining)]
        global_ratio = float(train_pool["label"].mean())
        val_patient = min(
            remaining,
            key=lambda patient: abs(
                float(metadata.loc[metadata["patient_id"] == patient, "label"].mean())
                - global_ratio
            ),
        )
        train_patients = [patient for patient in remaining if patient != val_patient]

        train_split = metadata[metadata["patient_id"].isin(train_patients)].copy()
        val_split = metadata[metadata["patient_id"] == val_patient].copy()
        test_split = metadata[metadata["patient_id"] == test_patient].copy()

        patient_sets = {
            "train": set(train_split["patient_id"].unique()),
            "validation": set(val_split["patient_id"].unique()),
            "test": set(test_split["patient_id"].unique()),
        }
        if patient_sets["train"] & patient_sets["validation"]:
            raise RuntimeError("Patient leakage between train and validation")
        if patient_sets["train"] & patient_sets["test"]:
            raise RuntimeError("Patient leakage between train and test")
        if patient_sets["validation"] & patient_sets["test"]:
            raise RuntimeError("Patient leakage between validation and test")

        yield {
            "test_patient": test_patient,
            "validation_patient": val_patient,
            "train_patients": train_patients,
            "train": train_split[OUTPUT_COLUMNS].reset_index(drop=True),
            "validation": val_split[OUTPUT_COLUMNS].reset_index(drop=True),
            "test": test_split[OUTPUT_COLUMNS].reset_index(drop=True),
        }


def save_fold_splits(fold_dir: Path, fold: dict) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold["train"].to_csv(fold_dir / "train_metadata.csv", index=False)
    fold["validation"].to_csv(fold_dir / "validation_metadata.csv", index=False)
    fold["validation"].to_csv(fold_dir / "val_metadata.csv", index=False)
    fold["test"].to_csv(fold_dir / "test_metadata.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--model", choices=["eegnet", "attention", "hybrid"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output-dir", default=str(experiment_path("cross_patient")))
    parser.add_argument("--splits-only", action="store_true")
    return parser.parse_args()


def main():
    ensure_project_storage()
    args = parse_args()
    overrides = {"model": args.model, "epochs": args.epochs}
    config = load_config(args.config or None, overrides)

    output_dir = artifact_path(args.output_dir, default_base=experiment_path())
    output_dir.mkdir(parents=True, exist_ok=True)
    print_storage_summary({"cross_patient_output_dir": output_dir})

    metadata = load_shard_metadata(discover_shards())
    summary = patient_summary(metadata)
    summary.to_csv(output_dir / "patient_summary.csv", index=False)
    print(summary.to_string(index=False))

    fold_results = []
    for fold_index, fold in enumerate(make_patient_folds(metadata, args.max_folds), start=1):
        fold_dir = output_dir / f"fold_{fold_index:02d}_{fold['test_patient']}"
        save_fold_splits(fold_dir, fold)

        print(
            f"\nFold {fold_index}: test={fold['test_patient']} "
            f"validation={fold['validation_patient']} "
            f"train={fold['train_patients']}"
        )

        if args.splits_only:
            fold_results.append(
                {
                    "fold": fold_index,
                    "test_patient": fold["test_patient"],
                    "validation_patient": fold["validation_patient"],
                    "train_patients": fold["train_patients"],
                    "status": "splits_only",
                }
            )
            continue

        fold_config = DEFAULT_CONFIG.copy()
        fold_config.update(config)
        fold_config["run_dir"] = str(fold_dir / "runs")
        fold_config["checkpoint_dir"] = str(fold_dir / "checkpoints")

        result = train_model(
            fold_config,
            train_meta=fold["train"],
            val_meta=fold["validation"],
            test_meta=fold["test"],
            run_name=f"fold_{fold_index:02d}_{fold['test_patient']}",
        )
        result.update(
            {
                "fold": fold_index,
                "test_patient": fold["test_patient"],
                "validation_patient": fold["validation_patient"],
                "train_patients": fold["train_patients"],
            }
        )
        fold_results.append(result)

    with open(output_dir / "cross_patient_results.json", "w", encoding="utf-8") as handle:
        json.dump(fold_results, handle, indent=2)
    print(f"\nSaved {output_dir / 'cross_patient_results.json'}")


if __name__ == "__main__":
    main()
