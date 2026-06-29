import mne
from channel_config import COMMON_CHANNELS

edf_file = "data/raw/chb01/chb01_01.edf"

raw = mne.io.read_raw_edf(
    edf_file,
    preload=False,
    verbose=False
)

print("\nChannels in EDF:")
print(raw.ch_names)

print("\nChecking channels:")

for ch in COMMON_CHANNELS:
    print(ch, "->", ch in raw.ch_names)