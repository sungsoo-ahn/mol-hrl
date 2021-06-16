import torch
from torch.nn.utils.rnn import pad_sequence

from vocabulary import PAD_TOKEN, smiles2seq, SmilesTokenizer, create_vocabulary
from util.mol import randomize_smiles
from util.mutate import mutate

PADDING_VALUE = 0

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, dir, tag, aug_randomize_smiles, aug_mutate):
        with open(f"{dir}/{tag}.txt", "r") as f:
            self.smiles_list =  f.read().splitlines()

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
        seqs = pad_sequence(seqs, batch_first=True, padding_value=self.vocab.get_pad_id())

        return seqs, lengths


class PairedSmilesDataset(SmilesDataset):
    def __getitem__(self, idx):
        smiles0 = smiles1 = self.smiles_list[idx]
        if self.transform is not None:
            smiles0 = self.transform(smiles0)
            smiles1 = self.transform(smiles1)

        seq0 = smiles2seq(smiles0, self.tokenizer, self.vocab)
        seq1 = smiles2seq(smiles1, self.tokenizer, self.vocab)

        return seq0, seq1


    def collate_fn(self, seqs01):
        seqs0, seqs1 = zip(*seqs01)
        lengths0 = torch.tensor([seq.size(0) for seq in seqs0])
        lengths1 = torch.tensor([seq.size(0) for seq in seqs1])
        seqs0 = pad_sequence(seqs0, batch_first=True, padding_value=self.vocab.get_pad_id())
        seqs1 = pad_sequence(seqs1, batch_first=True, padding_value=self.vocab.get_pad_id())

        return (seqs0, lengths0), (seqs1, lengths1)
