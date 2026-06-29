import sys
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM_DIR = Path("data/interim/windows")
PROCESSED_DIR = Path("data/processed")
X_OUT_PATH = PROCESSED_DIR / "X.npy"
Y_OUT_PATH = PROCESSED_DIR / "y.npy"
METADATA_OUT_PATH = PROCESSED_DIR / "metadata.csv"

COPY_CHUNK_SIZE = 512


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def discover_shards() -> list[tuple[Path, Path, Path]]:
    if not INTERIM_DIR.exists():
        fail(f"Shard directory does not exist: {INTERIM_DIR}")

    x_files = sorted(INTERIM_DIR.glob("*_X.npy"))
    if not x_files:
        fail(f"No *_X.npy shard files found in {INTERIM_DIR}")

    shards = []
    for x_path in x_files:
        edf_name = x_path.name[: -len("_X.npy")]
        y_path = INTERIM_DIR / f"{edf_name}_y.npy"
        meta_path = INTERIM_DIR / f"{edf_name}_meta.csv"

        missing = [path for path in (y_path, meta_path) if not path.exists()]
        if missing:
            fail(
                f"Missing shard companion(s) for {x_path.name}: "
                f"{[str(path) for path in missing]}"
            )

        shards.append((x_path, y_path, meta_path))

    return shards


def count_metadata_rows(meta_path: Path) -> int:
    total_rows = 0
    for chunk in pd.read_csv(meta_path, chunksize=100_000):
        total_rows += len(chunk)
    return total_rows


def validate_and_measure_shards(
    shards: list[tuple[Path, Path, Path]]
) -> tuple[int, tuple[int, ...], np.dtype, np.dtype, list[str]]:
    total_windows = 0
    feature_shape = None
    x_dtype = None
    y_dtype = None
    metadata_columns = None

    for shard_index, (x_path, y_path, meta_path) in enumerate(shards, start=1):
        try:
            X = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
        except Exception as exc:
            fail(f"Could not read shard {x_path.name}: {type(exc).__name__}: {exc}")

        if X.ndim != 3:
            fail(f"{x_path.name} must be 3D, got shape {X.shape}")
        if y.ndim != 1:
            fail(f"{y_path.name} must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            fail(
                f"Row mismatch in {x_path.name}: X has {X.shape[0]} rows, "
                f"y has {y.shape[0]} rows"
            )

        try:
            meta_header = pd.read_csv(meta_path, nrows=0)
            metadata_rows = count_metadata_rows(meta_path)
        except Exception as exc:
            fail(f"Could not read metadata {meta_path.name}: {type(exc).__name__}: {exc}")

        if metadata_rows != X.shape[0]:
            fail(
                f"Row mismatch in {meta_path.name}: metadata has {metadata_rows} rows, "
                f"X has {X.shape[0]} rows"
            )

        if feature_shape is None:
            feature_shape = tuple(X.shape[1:])
            x_dtype = X.dtype
            y_dtype = y.dtype
            metadata_columns = list(meta_header.columns)
        else:
            if tuple(X.shape[1:]) != feature_shape:
                fail(
                    f"Feature shape mismatch in {x_path.name}: expected "
                    f"{feature_shape}, got {tuple(X.shape[1:])}"
                )
            if X.dtype != x_dtype:
                fail(f"X dtype mismatch in {x_path.name}: expected {x_dtype}, got {X.dtype}")
            if y.dtype != y_dtype:
                fail(f"y dtype mismatch in {y_path.name}: expected {y_dtype}, got {y.dtype}")
            if list(meta_header.columns) != metadata_columns:
                fail(
                    f"Metadata columns mismatch in {meta_path.name}: expected "
                    f"{metadata_columns}, got {list(meta_header.columns)}"
                )

        total_windows += int(X.shape[0])
        print(
            f"[check {shard_index}/{len(shards)}] {x_path.name}: "
            f"{X.shape[0]} rows",
            flush=True,
        )

        del X
        del y

    if feature_shape is None or x_dtype is None or y_dtype is None or metadata_columns is None:
        fail("No valid shards found")

    return (
        total_windows,
        feature_shape,
        np.dtype(x_dtype),
        np.dtype(y_dtype),
        metadata_columns,
    )


def copy_arrays(
    shards: list[tuple[Path, Path, Path]],
    total_windows: int,
    feature_shape: tuple[int, ...],
    x_dtype: np.dtype,
    y_dtype: np.dtype,
) -> None:
    X_out = np.lib.format.open_memmap(
        X_OUT_PATH,
        mode="w+",
        dtype=x_dtype,
        shape=(total_windows, *feature_shape),
    )
    y_out = np.lib.format.open_memmap(
        Y_OUT_PATH,
        mode="w+",
        dtype=y_dtype,
        shape=(total_windows,),
    )

    offset = 0
    try:
        for shard_index, (x_path, y_path, _) in enumerate(shards, start=1):
            X = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            shard_rows = int(X.shape[0])

            for start in range(0, shard_rows, COPY_CHUNK_SIZE):
                end = min(start + COPY_CHUNK_SIZE, shard_rows)
                X_out[offset + start : offset + end] = X[start:end]
                y_out[offset + start : offset + end] = y[start:end]

            offset += shard_rows
            print(
                f"[merge {shard_index}/{len(shards)}] copied {x_path.name} "
                f"({shard_rows} rows, total={offset}/{total_windows})",
                flush=True,
            )

            del X
            del y
    finally:
        X_out.flush()
        y_out.flush()
        del X_out
        del y_out

    if offset != total_windows:
        fail(f"Copied {offset} rows, expected {total_windows}")


def copy_metadata(
    shards: list[tuple[Path, Path, Path]],
    expected_rows: int,
    metadata_columns: list[str],
) -> None:
    metadata_parts = []
    total_rows = 0

    for shard_index, (_, _, meta_path) in enumerate(shards, start=1):
        metadata = pd.read_csv(meta_path)
        if list(metadata.columns) != metadata_columns:
            fail(
                f"Metadata columns changed in {meta_path.name}: expected "
                f"{metadata_columns}, got {list(metadata.columns)}"
            )

        metadata_parts.append(metadata)
        total_rows += len(metadata)
        print(
            f"[metadata {shard_index}/{len(shards)}] read {meta_path.name} "
            f"({len(metadata)} rows)",
            flush=True,
        )

    if total_rows != expected_rows:
        fail(f"Metadata rows total {total_rows}, expected {expected_rows}")

    merged_metadata = pd.concat(metadata_parts, ignore_index=True)
    merged_metadata.to_csv(METADATA_OUT_PATH, index=False)

    del metadata_parts
    del merged_metadata


def verify_outputs(expected_rows: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    X = np.load(X_OUT_PATH, mmap_mode="r")
    y = np.load(Y_OUT_PATH, mmap_mode="r")
    metadata_rows = count_metadata_rows(METADATA_OUT_PATH)

    if X.shape[0] != expected_rows:
        fail(f"Final X rows {X.shape[0]}, expected {expected_rows}")
    if y.shape[0] != expected_rows:
        fail(f"Final y rows {y.shape[0]}, expected {expected_rows}")
    if metadata_rows != expected_rows:
        fail(f"Final metadata rows {metadata_rows}, expected {expected_rows}")

    X_shape = tuple(X.shape)
    y_shape = tuple(y.shape)
    del X
    del y
    return X_shape, y_shape, metadata_rows


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    shards = discover_shards()
    print(f"Found {len(shards)} EDF shard(s)", flush=True)

    total_windows, feature_shape, x_dtype, y_dtype, metadata_columns = (
        validate_and_measure_shards(shards)
    )
    print(f"Total windows: {total_windows}", flush=True)
    print(f"Feature shape: {feature_shape}", flush=True)
    print(f"X dtype: {x_dtype}", flush=True)
    print(f"y dtype: {y_dtype}", flush=True)

    copy_arrays(shards, total_windows, feature_shape, x_dtype, y_dtype)
    copy_metadata(shards, total_windows, metadata_columns)

    X_shape, y_shape, metadata_rows = verify_outputs(total_windows)
    print("Final dataset written", flush=True)
    print(f"X shape: {X_shape}", flush=True)
    print(f"y shape: {y_shape}", flush=True)
    print(f"metadata rows: {metadata_rows}", flush=True)
    print(f"Saved {X_OUT_PATH}", flush=True)
    print(f"Saved {Y_OUT_PATH}", flush=True)
    print(f"Saved {METADATA_OUT_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc
