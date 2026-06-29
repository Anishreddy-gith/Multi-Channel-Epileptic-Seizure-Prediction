import mne
from pathlib import Path

raw_dir = Path("data/raw")

for edf in raw_dir.rglob("*.edf"):
    try:
        raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)

        print("\n" + "=" * 60)
        print(edf.name)
        print("Channels:", len(raw.ch_names))
        print("Sampling Rate:", raw.info["sfreq"])

    except Exception as e:
        print(f"ERROR: {edf.name}")
        print(e)