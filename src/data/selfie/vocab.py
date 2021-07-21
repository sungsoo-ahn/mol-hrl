"""
Vocabulary helper class
"""

from data.smiles.util import load_smiles_list
import numpy as np
import torch
import selfies as sf

START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
MASK_TOKEN = "<MASK>"
START_ID = 2
END_ID = 1
PAD_ID = 0
MASK_ID = 3

class SelfieVocabulary:
    def __init__(self):
        self._tokens = dict()
        self._current_id = 0
        self.update([PAD_TOKEN, END_TOKEN, START_TOKEN, MASK_TOKEN])
        
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


class SelfieTokenizer:
    def tokenize(self, selfie):
        tokens = [i for i in sf.split_selfies(selfie)]
        tokens = [START_TOKEN] + tokens + [END_TOKEN]
        return tokens

    def untokenize(self, tokens):
        if tokens[0] != START_TOKEN:
            return ""
        
        if tokens[-1] != END_TOKEN:
            return "".join(tokens[1:])

        return "".join(tokens[1:-1])

def create_selfie_vocabulary(smiles_list, tokenizer):
    tokens = set()
    for smi in smiles_list:
        selfie = sf.encoder(smi)
        cur_tokens = tokenizer.tokenize(selfie)
        tokens.update(cur_tokens)

    vocabulary = SelfieVocabulary()
    vocabulary.update(sorted(tokens))
    return vocabulary

def load_selfie_tokenizer(data_dir):
    return SelfieTokenizer()

def load_selfie_vocabulary(data_dir):
    tokenizer = load_selfie_tokenizer(data_dir)
    smiles_list = load_smiles_list(data_dir, "full")
    vocabulary = create_selfie_vocabulary(smiles_list, tokenizer)
    return vocabulary

def selfie_sequence_from_smiles(smiles, tokenizer, vocabulary):
    selfie = sf.encoder(smiles)
    return torch.tensor(vocabulary.encode(tokenizer.tokenize(selfie)))

def smiles_from_selfie_sequence(sequence, tokenizer, vocabulary):
    selfie = tokenizer.untokenize(vocabulary.decode(sequence.squeeze(0).tolist()))
    smiles = sf.decoder(selfie)
    return smiles