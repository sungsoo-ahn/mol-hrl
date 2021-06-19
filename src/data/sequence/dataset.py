import torch
from torch.nn.utils.rnn import pad_sequence


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, handler):
        super(SequenceDataset, self).__init__()
        self.smiles_list = smiles_list
        self.handler = handler

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        sequence = self.handler.sequence_from_string(smiles)
        length = sequence.size(0)

        return sequence, torch.tensor(length)

    def __len__(self):
        return len(self.smiles_list)

    def collate_fn(self, sequence_list):
        sequences, lengths = zip(*sequence_list)
        sequences = pad_sequence(
            sequences, batch_first=True, padding_value=self.handler.vocabulary.get_pad_id()
        )
        lengths = torch.stack(lengths)
        batched_sequence_data = (sequences, lengths)
        return batched_sequence_data
