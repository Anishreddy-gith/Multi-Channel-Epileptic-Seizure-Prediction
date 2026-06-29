from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

BASE = "https://physionet.org/files/chbmit/1.0.0"
OUT = Path("data/raw")

PATIENTS = {
    "chb02": {
        "edf": [*[f"chb02_{i:02d}.edf" for i in range(1, 16)], "chb02_16+.edf", "chb02_16.edf", *[f"chb02_{i:02d}.edf" for i in range(17, 36)]],
        "ann": ["chb02_01.edf.seizures", "chb02_02.edf.seizures", "chb02_03.edf.seizures", "chb02_04.edf.seizures", "chb02_16+.edf.seizures", "chb02_16.edf.seizures", "chb02_19.edf.seizures"],
    },
    "chb03": {
        "edf": [f"chb03_{i:02d}.edf" for i in range(1, 39)],
        "ann": ["chb03_01.edf.seizures", "chb03_02.edf.seizures", "chb03_03.edf.seizures", "chb03_04.edf.seizures", "chb03_34.edf.seizures", "chb03_35.edf.seizures", "chb03_36.edf.seizures"],
    },
    "chb07": {
        "edf": [f"chb07_{i:02d}.edf" for i in range(1, 20)],
        "ann": ["chb07_12.edf.seizures", "chb07_13.edf.seizures", "chb07_19.edf.seizures"],
    },
    "chb09": {
        "edf": [f"chb09_{i:02d}.edf" for i in range(1, 20)],
        "ann": ["chb09_06.edf.seizures", "chb09_08.edf.seizures", "chb09_19.edf.seizures"],
    },
    "chb10": {
        "edf": [*[f"chb10_{i:02d}.edf" for i in range(1, 9)], *[f"chb10_{i:02d}.edf" for i in range(12, 23)], "chb10_27.edf", "chb10_28.edf", "chb10_30.edf", "chb10_31.edf", "chb10_38.edf", "chb10_89.edf"],
        "ann": ["chb10_12.edf.seizures", "chb10_20.edf.seizures", "chb10_27.edf.seizures", "chb10_30.edf.seizures", "chb10_31.edf.seizures", "chb10_38.edf.seizures", "chb10_89.edf.seizures"],
    },
}

def download_file(patient: str, filename: str) -> None:
    target_dir = OUT / patient
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{patient}/{quote(filename)}"
    target = target_dir / filename
    if target.exists():
        print(f"skip {patient}/{filename}")
        return
    print(f"download {patient}/{filename}")
    urlretrieve(url, target)

for patient, spec in PATIENTS.items():
    for filename in spec["edf"] + spec["ann"]:
        download_file(patient, filename)

print("done")