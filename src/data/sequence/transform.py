import torch
from data.sequence.vocab import MASK_ID


def mask_sequence(sequence, mask_rate):
    mask = torch.bernoulli(torch.full((sequence.size(0),), mask_rate)).bool()
    mask[0] = False
    mask[-1] = False
    sequence[mask] = MASK_ID
    return sequence
