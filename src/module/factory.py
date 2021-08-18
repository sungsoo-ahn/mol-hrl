from module.encoder.gnn import GNNEncoder
from module.decoder.lstm import LSTMDecoder
from module.decoder.transformer import TransformerDecoder

def load_decoder(decoder_name, code_dim):
    if decoder_name == "lstm_base":
        return LSTMDecoder(decoder_num_layers=3, decoder_hidden_dim=1024, code_dim=code_dim)
    
    elif decoder_name == "transformer_base":
        return TransformerDecoder(
            num_encoder_layers=6,
            emb_size=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            code_dim=code_dim,
            )
    
def load_encoder(encoder_name, code_dim):
    return GNNEncoder(encoder_num_layers=5, encoder_hidden_dim=256, code_dim=code_dim)
    