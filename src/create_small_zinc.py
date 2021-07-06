from data.smiles.util import load_smiles_list

smiles_list = load_smiles_list("../resource/data/zinc/", "full")
smiles_list = smiles_list[:50000]
with open("../resource/data/zinc_small/smiles_list.txt", "w") as f:
    f.write("\n".join(smiles_list) + "\n")
