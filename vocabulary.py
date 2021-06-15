"""
Vocabulary helper class
"""

import re
import numpy as np
import torch

START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

# contains the data structure
class Vocabulary:
    """Stores the tokens and their conversion to vocabulary indexes."""

    def __init__(self, max_length):
        self._tokens = dict()
        self._current_id = 0
        self.max_length = max_length
        self.update([PAD_TOKEN, END_TOKEN, START_TOKEN])

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
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
        """Encodes a list of tokens as vocabulary indexes."""
        vocab_index = np.zeros(len(tokens), dtype=np.int)
        for i, token in enumerate(tokens):
            vocab_index[i] = self._tokens[token]
        return vocab_index

    def decode(self, vocab_index):
        """Decodes a vocabulary index matrix to a list of tokens."""
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
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]

    def get_start_id(self):
        return self._tokens[START_TOKEN]
    
    def get_end_id(self):
        return self._tokens[END_TOKEN]

    def get_pad_id(self):
        return self._tokens[PAD_TOKEN]
    

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

def create_vocabulary(smiles_list, tokenizer, max_length):
    """Creates a vocabulary for the Smiles syntax."""
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi))

    vocabulary = Vocabulary(max_length)
    vocabulary.update(sorted(tokens))
    return vocabulary

def smiles2seq(smiles, tokenizer, vocabulary):
    return torch.tensor(vocabulary.encode(tokenizer.tokenize(smiles)))

def seq2smiles(seq, tokenizer, vocabulary):
    strings = []
    seqs = seq.cpu().split(1, dim=0)
    for seq in seqs:
        strings.append(tokenizer.untokenize(vocabulary.decode(seq.squeeze(0).tolist())))
    
    return strings

if __name__ == "__main__":
    tokenizer = SmilesTokenizer()
    vocab = create_vocabulary(["CCCCBBrB", "NNNN"], tokenizer)