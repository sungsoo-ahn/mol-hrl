from data.util import load_tokenizer
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import math


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def compute_accs(logits, tgt):
    batch_size = tgt.size(0)
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == tgt)
    correct[tgt == 0] = True

    acc_elem = correct[tgt != 0].float().mean()
    acc_seq = correct.view(batch_size, -1).all(dim=0).float().mean()

    return acc_elem, acc_seq

# Seq2Seq Network
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
        code_dim,
    ):
        super(TransformerDecoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = LayerNorm(emb_size)
        self.transformer = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.tokenizer = load_tokenizer()
        vocab_size = self.tokenizer.get_vocab_size()
        self.generator = nn.Linear(emb_size, vocab_size)
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.code_encoder = nn.Linear(code_dim, emb_size)
        
    def forward(self, batched_sequence_data, codes):
        batched_sequence_data = batched_sequence_data.transpose(0, 1)

        mask, key_padding_mask = self.create_mask(batched_sequence_data)
        outs = self.positional_encoding(self.tok_emb(batched_sequence_data)) + self.code_encoder(codes).unsqueeze(0)
        outs = self.transformer(outs, mask, key_padding_mask)
        logits = self.generator(outs)

        logits = logits.transpose(0, 1)
        return logits
    
    def sample(self, codes, argmax, max_len):
        batch_size = codes.size(0)
        ys = torch.ones(batch_size, 1).fill_(self.tokenizer.token_to_id("[BOS]")).type(torch.long).to(codes.device)
        
        ended = torch.zeros(batch_size, 1, dtype=torch.bool, device=codes.device)
        for _ in range(max_len-1):
            prob = self(ys, codes)[:, -1]
            if argmax:
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
            else:
                assert False

            next_word[ended] = self.tokenizer.token_to_id("[PAD]")
            ys = torch.cat([ys, next_word], dim=1)
            ended = ended | (next_word == self.tokenizer.token_to_id("[EOS]"))
            
        return ys

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, seq):
        seq_len = seq.shape[0]
        mask = self.generate_square_subsequent_mask(seq_len, device=seq.device)
        key_padding_mask = (seq == self.tokenizer.token_to_id("[PAD]")).transpose(0, 1)
        return mask, key_padding_mask    

