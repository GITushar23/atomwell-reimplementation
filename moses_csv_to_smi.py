# moses_csv_to_smi.py
import pandas as pd

df = pd.read_csv("data/molecules/moses_dataset_v1.csv")
train_smiles = df[df["SPLIT"] == "train"]["SMILES"].dropna().tolist()

with open("data/molecules/moses_train.smi", "w") as f:
    for s in train_smiles:
        f.write(s + "\n")
