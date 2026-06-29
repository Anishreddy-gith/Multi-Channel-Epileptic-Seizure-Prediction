from pathlib import Path
import struct

f = Path("data/raw/chb01/chb01_04.edf.seizures")

with open(f, "rb") as fp:
    data = fp.read()

print("File size:", len(data))
print("Raw bytes:", data)

print("\nIntegers:")
for i in range(0, len(data), 4):
    chunk = data[i:i+4]
    if len(chunk) == 4:
        print(i, struct.unpack("<I", chunk)[0])