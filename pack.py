#!/usr/bin/env python3
import numpy as np
from pathlib import Path

DATA_ROOT = Path("/teamspace/lightning_storage/bindwelldata/data")

VOCAB = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "[SMI]": 3, "[PROT]": 4, "[DNA]": 5}
SMILES_START, PROTEIN_START, DNA_START = 10, 50, 100

AA = "ACDEFGHIKLMNPQRSTVWY"
for i, aa in enumerate(AA):
    VOCAB[f"PROT_{aa}"] = PROTEIN_START + i

BASES = "ACGTN"
for i, base in enumerate(BASES):
    VOCAB[f"DNA_{base}"] = DNA_START + i

smiles_file = DATA_ROOT / "molecules" / "SMILES_random_50M.txt"
chars = set()
with open(smiles_file, 'r') as f:
    for line in f:
        chars.update(line.strip())

for i, ch in enumerate(sorted(chars)):
    VOCAB[f"SMI_{ch}"] = SMILES_START + i

np.save(DATA_ROOT / "vocab.npy", VOCAB)
print(f"Vocab size: {len(VOCAB)}")
print(f"Max ID: {max(VOCAB.values())}")
print("DNA tokens:", {k: v for k, v in VOCAB.items() if k.startswith("DNA_")})
