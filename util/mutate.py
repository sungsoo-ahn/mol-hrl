import random

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

#rdBase.DisableLog("rdApp.error")
RDLogger.DisableLog('rdApp.*')

def delete_atom():
    choices = [
        "[*:1]~[D1:2]>>[*:1]",
        "[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]",
        "[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]",
        "[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]",
        "[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]",
    ]
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return np.random.choice(choices, p=p)


def append_atom():
    choices = [
        ["single", ["C", "N", "O", "F", "S", "Cl", "Br"], 7 * [1.0 / 7.0]],
        ["double", ["C", "N", "O"], 3 * [1.0 / 3.0]],
        ["triple", ["C", "N"], 2 * [1.0 / 2.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == "single":
        rxn_smarts = "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)
    if BO == "double":
        rxn_smarts = "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)
    if BO == "triple":
        rxn_smarts = "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)

    return rxn_smarts


def insert_atom():
    choices = [
        ["single", ["C", "N", "O", "S"], 4 * [1.0 / 4.0]],
        ["double", ["C", "N"], 2 * [1.0 / 2.0]],
        ["triple", ["C"], [1.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == "single":
        rxn_smarts = "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)
    if BO == "double":
        rxn_smarts = "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)
    if BO == "triple":
        rxn_smarts = "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)

    return rxn_smarts


def change_bond_order():
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    p = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=p)


def delete_cyclic_bond():
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def add_ring():
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    p = [0.05, 0.05, 0.45, 0.45]

    return np.random.choice(choices, p=p)

RXN_SMARTS_LIST = []
RXN_SMARTS_LIST.append(None)
RXN_SMARTS_LIST.append(insert_atom())
RXN_SMARTS_LIST.append(change_bond_order())
RXN_SMARTS_LIST.append(delete_cyclic_bond())
RXN_SMARTS_LIST.append(add_ring())
RXN_SMARTS_LIST.append(delete_atom())
RXN_SMARTS_LIST.append(append_atom())
        

def change_atom(mol):
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    X = np.random.choice(choices, p=p)
    while not mol.HasSubstructMatch(Chem.MolFromSmarts("[" + X + "]")):
        X = np.random.choice(choices, p=p)
    Y = np.random.choice(choices, p=p)
    while Y == X:
        Y = np.random.choice(choices, p=p)

    return "[X:1]>>[Y:1]".replace("X", X).replace("Y", Y)

def ring_OK(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


# TODO: set from main? calculate for dataset?
average_size = 39.15
size_stdev = 3.50

def mol_ok(mol):
    try:
        Chem.SanitizeMol(mol)
        target_size = size_stdev * np.random.randn() + average_size  # parameters set in GA_mol
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except:
        return False

def mutate(smiles):
    mol = Chem.MolFromSmiles(smiles)
    #try:
    #    Chem.Kekulize(mol, clearAromaticFlags=True)
    #except ValueError:
    #    return smiles

    rxn_smarts = np.random.choice(RXN_SMARTS_LIST)
    if rxn_smarts is None:
        return smiles
    
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)

    new_mol_trial = rxn.RunReactants((mol,))

    new_mols = []
    for m in new_mol_trial:
        m = m[0]
        # print Chem.MolToSmiles(mol),mol_ok(mol)
        if mol_ok(m) and ring_OK(m):
            new_mols.append(m)

    if len(new_mols) > 0:
        return Chem.MolToSmiles(random.choice(new_mols))

    return smiles

if __name__ == "__main__":
    print(mutate("CCCCCCCCCC"))