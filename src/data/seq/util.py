"""
Vocabulary helper class
"""

from data.smiles.util import load_smiles_list
import re
import numpy as np
import torch

START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
MASK_TOKEN = "<MASK>"
START_ID = 2
END_ID = 1
PAD_ID = 0
MASK_ID = 3

class Vocabulary:
    def __init__(self):
        self._tokens = dict()
        self._current_id = 0
        self.update([PAD_TOKEN, END_TOKEN, START_TOKEN, MASK_TOKEN])
        self.update(["p"])

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        return self._tokens == other_vocabulary._tokens  # pylint: disable=W0212

    def __len__(self):
        return len(self._tokens) // 2

    def encode(self, tokens):
        vocab_index = np.zeros(len(tokens), dtype=np.int)
        for i, token in enumerate(tokens):
            vocab_index[i] = self._tokens[token]
        return vocab_index

    def decode(self, vocab_index):
        tokens = []
        for idx in vocab_index:
            token = self[idx]
            tokens.append(token)
            if token == END_TOKEN:
                break

        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        return [t for t in self._tokens if isinstance(t, str)]


class SmilesTokenizer:
    REGEXP = re.compile(
        "(\[|\]|Br?|Cl?|Si?|Se?|se?|@@?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    def tokenize(self, data):
        tokens = self.REGEXP.split(data)
        tokens = tokens[1::2]
        tokens = [START_TOKEN] + tokens + [END_TOKEN]

        return tokens

    def untokenize(self, tokens):
        if tokens[0] != START_TOKEN or tokens[-1] != END_TOKEN:
            return ""

        return "".join(tokens[1:-1])

def create_vocabulary(smiles_list, tokenizer):
    tokens = set()
    for smi in smiles_list:
        cur_tokens = tokenizer.tokenize(smi)
        tokens.update(cur_tokens)

    vocabulary = Vocabulary()
    vocabulary.update(sorted(tokens))
    return vocabulary

def load_tokenizer(data_dir):
    return SmilesTokenizer()

def load_vocabulary(data_dir):
    tokenizer = load_tokenizer(data_dir)
    smiles_list = load_smiles_list(data_dir, "full")
    vocabulary = create_vocabulary(smiles_list, tokenizer)
    return vocabulary

def sequence_from_string(string, tokenizer, vocabulary):
    return torch.tensor(vocabulary.encode(tokenizer.tokenize(string)))


def string_from_sequence(sequence, tokenizer, vocabulary):
    return tokenizer.untokenize(vocabulary.decode(sequence.squeeze(0).tolist()))
