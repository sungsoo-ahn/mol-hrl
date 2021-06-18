from functools import total_ordering
import numpy as np
import random


@total_ordering
class StorageElement:
    def __init__(self, smi, seq, length, score):
        self.smi = smi
        self.seq = seq
        self.length = length
        self.score = score

    def __eq__(self, other):
        return np.isclose(self.score, other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __hash__(self):
        return hash(self.smi)


def unravel_elems(elems):
    return tuple(
        map(
            list,
            zip(*[(elem.smi, elem.seq, elem.length, elem.score) for elem in elems]),
        )
    )


class MaxRewardPriorityQueue:
    def __init__(self, size):
        self.elems = []
        self.size = size

    def __len__(self):
        return len(self.elems)

    def add_list(self, smis, seqs, lengths, scores):
        new_elems = [
            StorageElement(smi=smi, seq=seq, length=length, score=score)
            for smi, seq, length, score in zip(smis, seqs, lengths, scores)
        ]
        self.elems.extend(new_elems)
        self.elems = list(set(self.elems))

        self.squeeze_by_kth(self.size)

    def get_elems(self):
        return unravel_elems(self.elems)

    def squeeze_by_kth(self, k):
        k = min(k, len(self.elems))
        self.elems = sorted(self.elems, reverse=True)[:k]
        
    def sample_batch(self, batch_size):
        sampled_elems = random.choices(population=self.elems, k=batch_size)
        return unravel_elems(sampled_elems)

class ReplayBuffer:
    def __init__(self):
        self.elems = []

    def __len__(self):
        return len(self.elems)

    def add_list(self, smis, seqs, lengths, scores):
        new_elems = [
            StorageElement(smi=smi, seq=seq, length=length, score=score)
            for smi, seq, length, score in zip(smis, seqs, lengths, scores)
        ]
        self.elems.extend(new_elems)
        if len(self.elems) > self.size:
            self.elems = self.elems[-self.size:]

    def sample_batch(self, batch_size):
        sampled_elems = random.choices(population=self.elems, k=batch_size)
        return unravel_elems(sampled_elems)
