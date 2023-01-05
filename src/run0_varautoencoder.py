import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
#from neptune.new.integrations.pytorch_lightning import NeptuneLogger
import torch
from pl_module.varautoencoder import VarAutoEncoderModule
from rdkit import Chem
from moses.metrics.metrics import get_all_metrics
BASE_CHECKPOINT_DIR = "../resource/checkpoint/run0_autoencoder"
def MolFromGraphs(node_list, adjacency_matrix):

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i].item())
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    
    for ix, row in enumerate(adjacency_matrix[0]):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            
            bond = bond.item()
            
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 4:
                #bond_type = Chem.rdchem.BondType.SINGLE
                bond_type = Chem.rdchem.BondType.AROMATIC
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()            

    return mol

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    VarAutoEncoderModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="../resource/checkpoint/default")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--decoder", type=str, default="lstm")
    parser.add_argument("--code_dim", type=int, default=16)
    parser.add_argument("--encoder_hidden_dim", type=int, default=512)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--resume_dir", type=str, default="")

    hparams = parser.parse_args()

    #neptune_logger = NeptuneLogger(project="sungsahn0215/molrep")
    #neptune_logger.run["params"] = vars(hparams)
    #neptune_logger.run['sys/tags'].add(["run0", "autoencoder"] + hparams.tag.split("_"))

    model = VarAutoEncoderModule(hparams)
    
    #checkpoint = torch.load(hparams.resume_dir)
    #model.load_state_dict(checkpoint["state_dict"])
    
            
    
    # with torch.no_grad():
    #   for i in range(len(model.val_dataset)):
    #       codes = model.encoder(model.val_dataset[i][0])
    #       mu = model.fc_mu(codes)
    #       torch.save(mu, f'latents_val/tensor_kekulize_mu_{i}.pt')
    #       logvar = model.fc_logvar(codes)
    #       torch.save(logvar, f'latents_val/tensor_kekulize_logvar_{i}.pt')
    # raise dd
    # smiles_list = []
    # model = model.cuda()
    # with torch.no_grad():
    #     for i in range(128):
    #         codes = torch.load(f"/home/osikjs/generated/latents_{i}.pt")
    #         codes = torch.Tensor(codes).cuda()
            
            
    #         h_atom, _, h_edge = model.decoder(codes, torch.Tensor([0]*codes.shape[0]).long().cuda())
    #         atoms = torch.argmax(h_atom, dim=-1)
    #         adjacency = torch.argmax(h_edge, dim=-1)

    #         print(atoms)
    #         print(adjacency)
    #         print(Chem.MolToSmiles(MolFromGraphs(atoms, adjacency)))
    #         smiles_list.append(Chem.MolToSmiles(MolFromGraphs(atoms, adjacency)))
    # train_smiles = []
    # test_smiles = []
    # with open("/home/osikjs/node-autoencoder/resource/data/zinc/train_full.txt", "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         train_smiles.append(line)
    # with open("/home/osikjs/node-autoencoder/resource/data/zinc/test.txt", "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         test_smiles.append(line)
    # scores = get_all_metrics(gen=smiles_list, k=len(smiles_list), device="cuda:0", n_jobs=8, test=test_smiles, train=train_smiles)
    # print(scores)
    # raise dd

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(BASE_CHECKPOINT_DIR, hparams.tag),
        monitor="validation/loss/edge_recon",
        filename="best",
        mode="min"
        )
    
    trainer = pl.Trainer(
        gpus=1,
        #logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)
