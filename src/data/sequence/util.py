import torch
from data.sequence.vocab import Vocabulary, SmilesTokenizer
from data.smiles.util import load_smiles_list


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


def smiles2sequence(string, tokenizer, vocabulary):
    return torch.LongTensor(vocabulary.encode(tokenizer.tokenize(string)))


def sequence2smiles(sequence, tokenizer, vocabulary):
    return tokenizer.untokenize(vocabulary.decode(sequence.squeeze(0).tolist()))
