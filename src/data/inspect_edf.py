import mne
from pathlib import Path

edf_file = Path("data/raw/chb01/chb01_01.edf")

print(f"Loading: {edf_file}")

raw = mne.io.read_raw_edf(edf_file, preload=False)

print("\n===== BASIC INFO =====")
print("Channels:", len(raw.ch_names))
print("Sampling Frequency:", raw.info["sfreq"])
print("Duration (seconds):", round(raw.times[-1], 2))

print("\n===== CHANNEL NAMES =====")
for ch in raw.ch_names:
    print(ch)