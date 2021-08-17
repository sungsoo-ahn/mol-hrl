import torch

def smiles2sequence(string, tokenizer, vocabulary):
    return vocabulary.encode(tokenizer.tokenize(string))

def sequence2smiles(sequence, tokenizer, vocabulary):
    return tokenizer.untokenize(vocabulary.decode(sequence))
