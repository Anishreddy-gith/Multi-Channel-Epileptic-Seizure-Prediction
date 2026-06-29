import itertools
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.sharded_dataset import SPLIT_DIR, discover_shards, load_shard_metadata

SEED = 42
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20

OUTPUT_COLUMNS = [
    "dataset_index",
    "shard_id",
    "shard_row",
    "patient_id",
    "label",
    "file_name",
]


def split_sizes(n_patients: int) -> tuple[int, int, int]:
    if n_patients < 3:
        raise ValueError("Need at least 3 patients for train/validation/test splits")

    train_size = max(1, int(round(TRAIN_FRACTION * n_patients)))
    validation_size = max(1, int(round(VALIDATION_FRACTION * n_patients)))
    test_size = n_patients - train_size - validation_size

    if test_size < 1:
        test_size = 1
        train_size = n_patients - validation_size - test_size
    if train_size < 1:
        raise ValueError("Training split would be empty")

    return train_size, validation_size, test_size


def score_split(metadata, train_patients, validation_patients, test_patients) -> float:
    global_ratio = float(metadata["label"].mean())
    total_rows = len(metadata)
    desired_rows = {
        "train": TRAIN_FRACTION * total_rows,
        "validation": VALIDATION_FRACTION * total_rows,
        "test": TEST_FRACTION * total_rows,
    }
    split_patients = {
        "train": train_patients,
        "validation": validation_patients,
        "test": test_patients,
    }

    score = 0.0
    for split_name, patients in split_patients.items():
        split = metadata[metadata["patient_id"].isin(patients)]
        if split.empty:
            return float("inf")
        if split["label"].nunique() < 2:
            score += 10.0

        score += abs(float(split["label"].mean()) - global_ratio)
        score += abs(len(split) - desired_rows[split_name]) / total_rows

    return score


def choose_patient_split(metadata):
    patients = sorted(metadata["patient_id"].unique())
    train_size, validation_size, test_size = split_sizes(len(patients))
    rng = np.random.default_rng(SEED)
    candidates = []

    for train_tuple in itertools.combinations(patients, train_size):
        remaining = [patient for patient in patients if patient not in train_tuple]
        for validation_tuple in itertools.combinations(remaining, validation_size):
            test_tuple = tuple(patient for patient in remaining if patient not in validation_tuple)
            if len(test_tuple) != test_size:
                continue

            train_patients = set(train_tuple)
            validation_patients = set(validation_tuple)
            test_patients = set(test_tuple)
            score = score_split(
                metadata,
                train_patients,
                validation_patients,
                test_patients,
            )
            candidates.append(
                (score, rng.random(), train_patients, validation_patients, test_patients)
            )

    if not candidates:
        raise RuntimeError("No valid patient split candidates found")

    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, train_patients, validation_patients, test_patients = candidates[0]
    return train_patients, validation_patients, test_patients


def build_split(metadata, patients):
    split = metadata[metadata["patient_id"].isin(patients)].copy()
    return split[OUTPUT_COLUMNS].sort_values("dataset_index").reset_index(drop=True)


def verify_splits(metadata, splits, patient_sets) -> None:
    for left, right in itertools.combinations(splits, 2):
        overlap = patient_sets[left].intersection(patient_sets[right])
        if overlap:
            raise RuntimeError(f"Patient leakage between {left} and {right}: {overlap}")

    combined = np.concatenate(
        [split["dataset_index"].to_numpy(dtype=np.int64) for split in splits.values()]
    )
    if len(combined) != len(set(combined.tolist())):
        raise RuntimeError("Duplicate dataset_index values across splits")
    if set(combined.tolist()) != set(metadata["dataset_index"].tolist()):
        raise RuntimeError("Splits do not cover all shard metadata rows exactly")

    for split_name, split in splits.items():
        if split.empty:
            raise RuntimeError(f"{split_name} split is empty")
        if split["label"].nunique() < 2:
            raise RuntimeError(f"{split_name} split does not contain both classes")


def print_summary(metadata, splits, patient_sets) -> None:
    print("Shard-native patient split")
    print("Whole patients are assigned to one split only; no merged arrays are used.")
    print(f"Random seed: {SEED}")
    print(f"Total windows: {len(metadata)}")
    print(f"Global positive ratio: {metadata['label'].mean():.4f}")

    for split_name, split in splits.items():
        counts = split["label"].value_counts().sort_index().to_dict()
        print(
            f"{split_name:10s} patients={sorted(patient_sets[split_name])} "
            f"rows={len(split)} labels={counts} "
            f"positive_ratio={split['label'].mean():.4f}"
        )


def save_splits(splits) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": SPLIT_DIR / "train_metadata.csv",
        "validation": SPLIT_DIR / "validation_metadata.csv",
        "test": SPLIT_DIR / "test_metadata.csv",
    }

    for split_name, split in splits.items():
        split.to_csv(paths[split_name], index=False)
        print(f"Saved {paths[split_name]}")

    val_alias = SPLIT_DIR / "val_metadata.csv"
    splits["validation"].to_csv(val_alias, index=False)
    print(f"Saved {val_alias} compatibility alias")


def main() -> None:
    shards = discover_shards()
    metadata = load_shard_metadata(shards)

    train_patients, validation_patients, test_patients = choose_patient_split(metadata)
    patient_sets = {
        "train": train_patients,
        "validation": validation_patients,
        "test": test_patients,
    }
    splits = {
        split_name: build_split(metadata, patients)
        for split_name, patients in patient_sets.items()
    }

    verify_splits(metadata, splits, patient_sets)
    print_summary(metadata, splits, patient_sets)
    save_splits(splits)
    print("Done")


if __name__ == "__main__":
    main()
