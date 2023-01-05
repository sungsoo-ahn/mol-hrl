from argparse import Namespace
from re import S
from data.util import load_tokenizer

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#from module.factory import load_encoder, load_decoder
from module.decoder.lstm import LSTMDecoder
from module.decoder.transformer import TransformerDecoder

from module.encoder.gnn import GNNEncoder
from module.vq_layer import FlattenedVectorQuantizeLayer
from pl_module.util import compute_sequence_cross_entropy, compute_sequence_accuracy
from data.factory import load_dataset, load_collate
from data.smiles.util import canonicalize



#encoder = GNNEncoder(
