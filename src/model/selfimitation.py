from collections import deque
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader

from net.gnn import GnnEncoder
from net.rnn import RnnDecoder
from data.sequence.handler import SequenceHandler
from data.pyg.handler import PyGHandler
from data.sequence.collate import collate_sequence_data_list
from data.pyg.collate import collate_pyg_data_list
from util.molecule.mol import is_valid_smiles
from util.sequence import compute_sequence_cross_entropy, compute_sequence_accuracy

class Buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)

    def append(self, element):
        self.buffer.append(element)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

class SourceDataset(IterableDataset):
    def __init__(self, generate_batch) -> None:
        self.generate_batch = generate_batch

    def __iter__(self):
        iterator = self.generate_batch()
        return iterator

class SelfImitationModel(pl.LightningModule):
    def __init__(
        self,
        encoder_num_layer,
        encoder_emb_dim,
        encoder_load_path,
        encoder_optimize,
        decoder_num_layers,
        decoder_hidden_dim,
        decoder_code_dim,
        decoder_load_path,
        decoder_optimize,
        data_dir,
        batch_size,
        batches_per_epoch,
    ):
        super(SelfImitationModel, self).__init__()
        # Encoder
        self.pyg_handler = PyGHandler()               
        self.encoder = GnnEncoder(num_layer=encoder_num_layer, emb_dim=encoder_emb_dim)
        if encoder_load_path != "":
            self.encoder.load_state_dict(torch.load(encoder_load_path))
        self.encoder_optimize = encoder_optimize
        if not self.encoder_optimize:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Decoder
        self.sequence_handler = SequenceHandler(data_dir)
        num_vocabs = len(self.sequence_handler.vocabulary)
        self.decoder = RnnDecoder(
            num_layers=decoder_num_layers,
            input_dim=num_vocabs,
            output_dim=num_vocabs,
            hidden_dim=decoder_hidden_dim,
            code_dim=decoder_code_dim,
        )
        if decoder_load_path != "":
            self.decoder.load_state_dict(torch.load(decoder_load_path))
        
        # Code
        self.decoder_code_dim = decoder_code_dim
        self.batch_size = batch_size
        
        # Optimization
        self.step = 0
        self.batches_per_epoch = batches_per_epoch
        
        self.save_hyperparameters()

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("imitation")

        # encoder parameters
        group.add_argument("--encoder_num_layer", type=int, default=5)
        group.add_argument("--encoder_emb_dim", type=int, default=300)
        group.add_argument("--encoder_optimize", action="store_true")
        group.add_argument("--encoder_load_path", type=str, default="")
        
        # decoder parameters
        group.add_argument("--decoder_num_layers", type=int, default=3)
        group.add_argument("--decoder_hidden_dim", type=int, default=1024)
        group.add_argument("--decoder_code_dim", type=int, default=300)
        group.add_argument("--decoder_load_path", type=str, default="")
        group.add_argument("--decoder_optimize", action="store_true")
        group.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
                        
        # optimizing parameters
        group.add_argument("--batches_per_epoch", type=int, default=1000)
        group.add_argument("--batch_size", type=int, default=256)

        return parser

    def train_dataloader(self):
        self.dataset = torch.randn(
            self.batch_size * self.batches_per_epoch, self.decoder_code_dim
            )
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)
                
    def training_step(self, batched_data, batch_idx):
        codes = batched_data
        with torch.no_grad():
            codes = torch.nn.functional.normalize(codes, p=2, dim=1)
            sequences, lengths, _ = self.decoder.sample(
                codes, 
                self.sequence_handler.vocabulary.get_start_id(), 
                self.sequence_handler.vocabulary.get_end_id(), 
                self.sequence_handler.vocabulary.get_max_length()
                )
        
        string_list = self.sequence_handler.strings_from_sequences(sequences, lengths)

        smiles_list, sequence_data_list, pyg_data_list = [], [], []
        valid_codes = []
        for idx, string in enumerate(string_list):
            if not is_valid_smiles(string):
                continue

            valid_codes.append(codes[idx])
            
            sequence = self.sequence_handler.sequence_from_string(string)
            sequence_data = (sequence, torch.tensor(sequence.size(0)))
            pyg_data = self.pyg_handler.pyg_from_string(string)

            smiles_list.append(string)
            sequence_data_list.append(sequence_data)
            pyg_data_list.append(pyg_data)
        
        valid_codes = torch.stack(valid_codes, dim=0)

        num_valid_samples = len(smiles_list)
        self.log("sample/valid_smiles_ratio", num_valid_samples / self.batch_size, prog_bar=True)

        batched_sequence_data = collate_sequence_data_list(
            sequence_data_list, self.sequence_handler.vocabulary.get_pad_id()
            )
        batched_pyg_data = collate_pyg_data_list(pyg_data_list)

        batched_sequence_data = (batched_sequence_data[0].to(self.device), batched_sequence_data[1].to(self.device))
        batched_pyg_data = batched_pyg_data.to(self.device)
        
        with torch.no_grad():
            relabeled_codes = self.encoder(batched_pyg_data)
            relabeled_codes = torch.nn.functional.normalize(relabeled_codes, p=2, dim=1)

        logits = self.decoder(batched_sequence_data, relabeled_codes)
        loss = compute_sequence_cross_entropy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        ) * num_valid_samples / self.batch_size
        elem_acc, sequence_acc = compute_sequence_accuracy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )

        code_mse = torch.nn.functional.mse_loss(valid_codes, relabeled_codes)

        self.log("train/loss/total", loss, on_step=True, logger=True)
        self.log("train/acc/element", elem_acc, on_step=True, logger=True)
        self.log("train/acc/sequence", sequence_acc, on_step=True, logger=True)
        self.log("train/stat/code_mse", code_mse, on_step=True, logger=True)

        return loss

    def configure_optimizers(self):
        params = list(self.decoder.parameters())
        if self.encoder_optimize:
            params += list(self.encoder.parameters())

        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]
