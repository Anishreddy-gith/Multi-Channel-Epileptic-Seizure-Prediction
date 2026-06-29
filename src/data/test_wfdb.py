import wfdb

ann = wfdb.rdann(
    "data/raw/chb01/chb01_04.edf",
    "seizures"
)

print("Samples:", ann.sample)
print("Symbols:", ann.symbol)
print("Aux Notes:", ann.aux_note)