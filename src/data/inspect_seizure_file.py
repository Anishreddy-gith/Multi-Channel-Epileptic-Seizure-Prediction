from pathlib import Path

f = Path("data/raw/chb01/chb01_04.edf.seizures")

with open(f, "rb") as fp:
    data = fp.read()

print("Size:", len(data))
print(data[:100])