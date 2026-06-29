from pathlib import Path

for ann in Path("data/raw").rglob("*.edf.seizures"):
    edf = Path(str(ann).replace(".seizures", ""))

    if not edf.exists():
        print("Missing EDF:", edf.name)