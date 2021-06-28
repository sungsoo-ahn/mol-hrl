import torch
from torch.nn.utils.rnn import pad_sequence


def collate_sequence_data_list(sequence_list, pad_id):
    sequences, lengths = zip(*sequence_list)
    sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
    lengths = torch.stack(lengths)
    batched_sequence_data = (sequences, lengths)
    return batched_sequence_data
