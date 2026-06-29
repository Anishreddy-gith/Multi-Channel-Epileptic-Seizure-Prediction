import gc
import shutil
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.channel_config import COMMON_CHANNELS

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim/windows")
SEIZURE_METADATA_PATH = PROCESSED_DIR / "seizure_metadata.csv"

WINDOW_SEC = 4
STRIDE_SEC = 2
PREICTAL_SEC = 10 * 60
POSTICTAL_GAP_SEC = 60

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
NOTCH_FREQ = 50.0

X_DTYPE = np.float32
Y_DTYPE = np.int64
EPS = 1e-8

META_COLUMNS = [
    "patient_id",
    "file_name",
    "segment_type",
    "window_start_sample",
    "window_end_sample",
    "window_start_sec",
    "window_end_sec",
    "label",
    "seizure_start_sec",
    "seizure_end_sec",
]


def load_seizure_metadata() -> pd.DataFrame:
    if not SEIZURE_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing seizure metadata: {SEIZURE_METADATA_PATH}")

    metadata = pd.read_csv(SEIZURE_METADATA_PATH)
    required_columns = {"file", "start_sec", "end_sec"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"{SEIZURE_METADATA_PATH} is missing columns: {sorted(missing_columns)}"
        )

    metadata = metadata.copy()
    metadata["file"] = metadata["file"].astype(str)
    metadata["start_sec"] = metadata["start_sec"].astype(float)
    metadata["end_sec"] = metadata["end_sec"].astype(float)
    return metadata


def find_edf_files() -> list[Path]:
    return sorted(RAW_DIR.glob("chb*/*.edf"))


def patient_id_from_path(edf_path: Path) -> str:
    return edf_path.parent.name


def output_paths(edf_path: Path) -> tuple[Path, Path, Path]:
    edf_name = edf_path.name
    x_suffix = "_X" + ".npy"
    y_suffix = "_y" + ".npy"
    meta_suffix = "_meta" + ".csv"
    return (
        INTERIM_DIR / f"{edf_name}{x_suffix}",
        INTERIM_DIR / f"{edf_name}{y_suffix}",
        INTERIM_DIR / f"{edf_name}{meta_suffix}",
    )


def remove_stale_shards(edf_path: Path) -> None:
    current_paths = list(output_paths(edf_path))
    legacy_stem = edf_path.stem
    x_suffix = "_X" + ".npy"
    y_suffix = "_y" + ".npy"
    meta_suffix = "_meta" + ".csv"
    legacy_paths = [
        INTERIM_DIR / f"{legacy_stem}{x_suffix}",
        INTERIM_DIR / f"{legacy_stem}{y_suffix}",
        INTERIM_DIR / f"{legacy_stem}{meta_suffix}",
    ]

    removed = []
    for path in [*current_paths, *legacy_paths]:
        if path.exists():
            path.unlink()
            removed.append(path.name)

    if removed:
        print(f"  Removed stale shards: {removed}", flush=True)


def get_ram_usage() -> str:
    try:
        import psutil

        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        total_gb = memory.total / (1024**3)
        return f"{used_gb:.2f}/{total_gb:.2f} GiB ({memory.percent:.1f}%)"
    except Exception:
        return "unavailable"


def get_free_disk_space() -> str:
    target = INTERIM_DIR if INTERIM_DIR.exists() else Path(".")
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    return f"{free_gb:.2f}/{total_gb:.2f} GiB free"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def matching_channel_names(raw: mne.io.BaseRaw) -> list[str] | None:
    available = {channel.strip(): channel for channel in raw.ch_names}
    missing = [channel for channel in COMMON_CHANNELS if channel not in available]
    if missing:
        return None
    return [available[channel] for channel in COMMON_CHANNELS]


def seizure_rows_for_file(seizure_metadata: pd.DataFrame, edf_name: str) -> pd.DataFrame:
    rows = seizure_metadata[seizure_metadata["file"] == edf_name].copy()
    if rows.empty:
        return rows
    return rows.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)


def add_interval(
    intervals: list[dict],
    start_sec: float,
    end_sec: float,
    label: int,
    segment_type: str,
    seizure_start_sec: float | None,
    seizure_end_sec: float | None,
) -> None:
    if end_sec <= start_sec:
        return
    intervals.append(
        {
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "label": int(label),
            "segment_type": segment_type,
            "seizure_start_sec": seizure_start_sec,
            "seizure_end_sec": seizure_end_sec,
        }
    )


def build_label_intervals(
    seizure_rows: pd.DataFrame,
    duration_sec: float,
) -> list[dict]:
    if seizure_rows.empty:
        return [
            {
                "start_sec": 0.0,
                "end_sec": float(duration_sec),
                "label": 0,
                "segment_type": "interictal_whole",
                "seizure_start_sec": None,
                "seizure_end_sec": None,
            }
        ]

    intervals: list[dict] = []
    cursor = 0.0
    seizures = [
        (float(row.start_sec), float(row.end_sec))
        for row in seizure_rows.itertuples(index=False)
        if float(row.end_sec) > float(row.start_sec)
    ]

    if not seizures:
        return [
            {
                "start_sec": 0.0,
                "end_sec": float(duration_sec),
                "label": 0,
                "segment_type": "interictal_whole",
                "seizure_start_sec": None,
                "seizure_end_sec": None,
            }
        ]

    for seizure_index, (seizure_start, seizure_end) in enumerate(seizures):
        seizure_start = min(max(0.0, seizure_start), duration_sec)
        seizure_end = min(max(seizure_start, seizure_end), duration_sec)
        preictal_start = max(0.0, seizure_start - PREICTAL_SEC, cursor)

        if preictal_start > cursor:
            if seizure_index == 0:
                segment_type = "interictal_before"
                context_start = seizure_start
                context_end = seizure_end
            else:
                segment_type = "interictal_after"
                context_start, context_end = seizures[seizure_index - 1]
            add_interval(
                intervals,
                cursor,
                preictal_start,
                0,
                segment_type,
                context_start,
                context_end,
            )

        add_interval(
            intervals,
            preictal_start,
            seizure_start,
            1,
            "preictal",
            seizure_start,
            seizure_end,
        )

        cursor = max(cursor, seizure_end + POSTICTAL_GAP_SEC)

    if cursor < duration_sec:
        context_start, context_end = seizures[-1]
        add_interval(
            intervals,
            cursor,
            duration_sec,
            0,
            "interictal_after",
            context_start,
            context_end,
        )

    return intervals


def interval_sample_bounds(
    interval: dict,
    sfreq: float,
    n_samples: int,
) -> tuple[int, int]:
    start_sample = max(0, int(np.ceil(interval["start_sec"] * sfreq)))
    end_sample = min(n_samples, int(np.floor(interval["end_sec"] * sfreq)))
    return start_sample, end_sample


def count_windows(intervals: list[dict], sfreq: float, n_samples: int) -> int:
    win_size = int(round(WINDOW_SEC * sfreq))
    stride = int(round(STRIDE_SEC * sfreq))
    total = 0

    for interval in intervals:
        start_sample, end_sample = interval_sample_bounds(interval, sfreq, n_samples)
        usable_samples = end_sample - start_sample
        if usable_samples >= win_size:
            total += ((usable_samples - win_size) // stride) + 1

    return int(total)


def normalize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return ((window - mean) / std).astype(X_DTYPE, copy=False)


def create_windows_for_edf(
    data: np.ndarray,
    sfreq: float,
    intervals: list[dict],
    patient_id: str,
    edf_name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    win_size = int(round(WINDOW_SEC * sfreq))
    stride = int(round(STRIDE_SEC * sfreq))
    n_windows = count_windows(intervals, sfreq, data.shape[1])

    X = np.empty((n_windows, len(COMMON_CHANNELS), win_size), dtype=X_DTYPE)
    y = np.empty((n_windows,), dtype=Y_DTYPE)
    meta: list[dict] = []

    out_index = 0
    for interval in intervals:
        start_sample, end_sample = interval_sample_bounds(
            interval,
            sfreq,
            data.shape[1],
        )
        if end_sample - start_sample < win_size:
            continue

        last_start = end_sample - win_size
        for window_start in range(start_sample, last_start + 1, stride):
            window_end = window_start + win_size
            label = int(interval["label"])
            X[out_index] = normalize_window(data[:, window_start:window_end])
            y[out_index] = label
            meta.append(
                {
                    "patient_id": patient_id,
                    "file_name": edf_name,
                    "segment_type": interval["segment_type"],
                    "window_start_sample": window_start,
                    "window_end_sample": window_end,
                    "window_start_sec": window_start / sfreq,
                    "window_end_sec": window_end / sfreq,
                    "label": label,
                    "seizure_start_sec": interval["seizure_start_sec"],
                    "seizure_end_sec": interval["seizure_end_sec"],
                }
            )
            out_index += 1

    if out_index != n_windows:
        raise RuntimeError(
            f"Window count mismatch for {edf_name}: counted {n_windows}, "
            f"created {out_index}"
        )

    return X, y, meta


def count_csv_rows(csv_path: Path) -> int:
    total_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        total_rows += len(chunk)
    return total_rows


def save_edf_shards(
    edf_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    meta: list[dict],
) -> None:
    if len(X) != len(y) or len(X) != len(meta):
        raise RuntimeError(
            f"Length mismatch for {edf_path.name}: "
            f"X={len(X)}, y={len(y)}, metadata={len(meta)}"
        )

    x_path, y_path, meta_path = output_paths(edf_path)
    np.save(x_path, X.astype(X_DTYPE, copy=False))
    np.save(y_path, y.astype(Y_DTYPE, copy=False))
    pd.DataFrame(meta, columns=META_COLUMNS).to_csv(meta_path, index=False)

    saved_X = np.load(x_path, mmap_mode="r")
    saved_y = np.load(y_path, mmap_mode="r")
    saved_meta_rows = count_csv_rows(meta_path)
    if saved_X.shape[0] != saved_y.shape[0] or saved_X.shape[0] != saved_meta_rows:
        raise RuntimeError(
            f"Saved length mismatch for {edf_path.name}: "
            f"X={saved_X.shape[0]}, y={saved_y.shape[0]}, metadata={saved_meta_rows}"
        )
    del saved_X
    del saved_y


def preprocess_one_edf(
    edf_path: Path,
    seizure_metadata: pd.DataFrame,
    edf_number: int,
    total_edfs: int,
) -> dict:
    start_time = time.perf_counter()
    patient_id = patient_id_from_path(edf_path)
    edf_name = edf_path.name

    raw = None
    data = None
    X = None
    y = None
    meta = None

    remove_stale_shards(edf_path)

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        selected_channels = matching_channel_names(raw)
        if selected_channels is None:
            available = {name.strip() for name in raw.ch_names}
            missing = [channel for channel in COMMON_CHANNELS if channel not in available]
            elapsed = time.perf_counter() - start_time
            print(
                f"[{edf_number}/{total_edfs}] patient={patient_id} file={edf_name} "
                f"SKIPPED missing_channels={missing} time={format_duration(elapsed)}",
                flush=True,
            )
            return {
                "status": "skipped",
                "windows": 0,
                "positive": 0,
                "negative": 0,
                "elapsed": elapsed,
            }

        raw.pick(selected_channels)
        raw.reorder_channels(selected_channels)
        raw.filter(
            l_freq=BANDPASS_LOW,
            h_freq=BANDPASS_HIGH,
            method="fir",
            fir_design="firwin",
            verbose=False,
        )
        raw.notch_filter(
            freqs=[NOTCH_FREQ],
            method="fir",
            fir_design="firwin",
            verbose=False,
        )

        sfreq = float(raw.info["sfreq"])
        data = raw.get_data().astype(X_DTYPE, copy=False)
        duration_sec = data.shape[1] / sfreq

        seizure_rows = seizure_rows_for_file(seizure_metadata, edf_name)
        intervals = build_label_intervals(seizure_rows, duration_sec)
        X, y, meta = create_windows_for_edf(
            data=data,
            sfreq=sfreq,
            intervals=intervals,
            patient_id=patient_id,
            edf_name=edf_name,
        )
        save_edf_shards(edf_path, X, y, meta)

        positive_windows = int(np.sum(y == 1))
        negative_windows = int(np.sum(y == 0))
        elapsed = time.perf_counter() - start_time
        print(
            f"[{edf_number}/{total_edfs}] patient={patient_id} file={edf_name} "
            f"windows={len(y)} positive={positive_windows} negative={negative_windows} "
            f"time={format_duration(elapsed)}",
            flush=True,
        )

        return {
            "status": "processed",
            "windows": int(len(y)),
            "positive": positive_windows,
            "negative": negative_windows,
            "elapsed": elapsed,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        remove_stale_shards(edf_path)
        print(
            f"[{edf_number}/{total_edfs}] patient={patient_id} file={edf_name} "
            f"FAILED error={type(exc).__name__}: {exc} time={format_duration(elapsed)}",
            flush=True,
        )
        return {
            "status": "failed",
            "windows": 0,
            "positive": 0,
            "negative": 0,
            "elapsed": elapsed,
        }

    finally:
        if raw is not None:
            del raw
        if data is not None:
            del data
        if X is not None:
            del X
        if y is not None:
            del y
        if meta is not None:
            del meta
        gc.collect()


def print_periodic_status(
    processed_count: int,
    total_count: int,
    total_elapsed: float,
) -> None:
    average_time = total_elapsed / processed_count
    remaining = total_count - processed_count
    estimated_remaining = average_time * remaining
    print(
        f"Status after {processed_count} EDFs: "
        f"RAM={get_ram_usage()} | disk={get_free_disk_space()} | "
        f"ETA={format_duration(estimated_remaining)}",
        flush=True,
    )


def main() -> None:
    mne.set_log_level("ERROR")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    seizure_metadata = load_seizure_metadata()
    edf_files = find_edf_files()
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found under {RAW_DIR}")

    print(f"Found {len(edf_files)} EDF files", flush=True)
    print(f"Writing per-EDF shards to {INTERIM_DIR}", flush=True)

    run_start = time.perf_counter()
    totals = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "windows": 0,
        "positive": 0,
        "negative": 0,
    }

    for edf_number, edf_path in enumerate(edf_files, start=1):
        result = preprocess_one_edf(
            edf_path=edf_path,
            seizure_metadata=seizure_metadata,
            edf_number=edf_number,
            total_edfs=len(edf_files),
        )
        totals[result["status"]] += 1
        totals["windows"] += result["windows"]
        totals["positive"] += result["positive"]
        totals["negative"] += result["negative"]

        if edf_number % 10 == 0:
            print_periodic_status(
                processed_count=edf_number,
                total_count=len(edf_files),
                total_elapsed=time.perf_counter() - run_start,
            )

    total_time = time.perf_counter() - run_start
    print(
        "Done. "
        f"processed={totals['processed']} skipped={totals['skipped']} "
        f"failed={totals['failed']} windows={totals['windows']} "
        f"positive={totals['positive']} negative={totals['negative']} "
        f"time={format_duration(total_time)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
