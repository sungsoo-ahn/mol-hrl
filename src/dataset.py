import torch
from torch.nn.utils.rnn import pad_sequence

from vocabulary import PAD_TOKEN, smiles2seq, SmilesTokenizer, create_vocabulary
from util.mol import randomize_smiles, smiles2graph
from util.mutate import mutate
from scoring.featurizer import compute_feature

import numpy as np

PADDING_VALUE = 0

def pyg_collate(batch):
    elem = batch[0]
    if isinstance(elem, Data):
        return Batch.from_data_list(batch, [], [])
    elif isinstance(elem, torch.Tensor):
        return default_collate(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: pyg_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return type(elem)(*(pyg_collate(s) for s in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        return [pyg_collate(s) for s in zip(*batch)]

    raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, dir, tag, aug_randomize_smiles, aug_mutate):
        with open(f"{dir}/{tag}.txt", "r") as f:
            self.smiles_list = f.read().splitlines()

        self.tokenizer = SmilesTokenizer()
        self.vocab = create_vocabulary(self.smiles_list, self.tokenizer)

        def transform(smiles):
            if aug_randomize_smiles:
                smiles = randomize_smiles(smiles)
            if aug_mutate:
                smiles = mutate(smiles)

            return smiles

        self.transform = transform

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("dataset")
        group.add_argument("--data_dir", type=str, default="../resource/data/")
        group.add_argument("--data_tag", type=str, default="zinc")
        group.add_argument("--data_aug_randomize_smiles", action="store_true")
        group.add_argument("--data_aug_mutate", action="store_true")
        return parser

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.transform is not None:
            smiles = self.transform(smiles)

        seq = smiles2seq(smiles, self.tokenizer, self.vocab)

        return seq

    def __len__(self):
        return len(self.smiles_list)

    def collate_fn(self, seqs):
        lengths = torch.tensor([seq.size(0) for seq in seqs])
        seqs = pad_sequence(
            seqs, batch_first=True, padding_value=self.vocab.get_pad_id()
        )

        return seqs, lengths

class FeaturizedSmilesDataset(SmilesDataset):
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.transform is not None:
            smiles = self.transform(smiles)

        seq = smiles2seq(smiles, self.tokenizer, self.vocab)

        feature = compute_feature(smiles)
        feature = torch.FloatTensor(np.concatenate([feature["int"], feature["float"], feature["fp"]], axis=0))

        return seq, feature

    def collate_fn(self, seqs_and_graphs_features):
        seqs, graphs, features = zip(*seqs_and_graphs_features)

        lengths = torch.tensor([seq.size(0) for seq in seqs])
        seqs = pad_sequence(
            seqs, batch_first=True, padding_value=self.vocab.get_pad_id()
        )
        features = torch.stack(features, dim=0)

        return (seqs, lengths), features