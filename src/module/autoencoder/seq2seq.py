import torch.nn as nn

from data.sequence.dataset import SequenceDataset
from module.encoder.sequence import SequenceEncoder
from module.decoder.sequence import SequenceDecoder
from module.autoencoder.base import BaseAutoEncoder


class Seq2SeqAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(BaseAutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = SequenceEncoder(hparams)
        self.decoder = SequenceDecoder(hparams)

    def get_input_dataset(self, split):
        return SequenceDataset(self.hparams.data_dir, split)

    def collate(self, data_list):
        input_data_list, target_data_list = zip(*data_list)
        batched_input_data = SequenceDataset.collate_fn(input_data_list)
        batched_target_data = SequenceDataset.collate_fn(target_data_list)
        return batched_input_data, batched_target_data
