import wfdb
import pandas as pd
from pathlib import Path

rows = []

for seizure_file in Path("data/raw").rglob("*.edf.seizures"):

    record = str(seizure_file).replace(".seizures", "")

    try:
        ann = wfdb.rdann(record, "seizures")

        sfreq = 256

        start_sample = int(ann.sample[0])
        end_sample = int(ann.sample[1])

        rows.append({
            "file": Path(record).name,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_sec": start_sample / sfreq,
            "end_sec": end_sample / sfreq
        })

    except Exception as e:
        print("ERROR:", seizure_file)
        print(e)

df = pd.DataFrame(rows)

print(df)

df.to_csv(
    "data/processed/seizure_metadata.csv",
    index=False
)

print("\nSaved:")
print("data/processed/seizure_metadata.csv")