from data.sequence.handler import SequenceHandler
import torch
import pytorch_lightning as pl

from net.gnn import GnnEncoder
from net.rnn import RnnDecoder
from data.sequence.handler import SequenceHandler
from util.sequence import compute_sequence_accuracy, compute_sequence_cross_entropy

class ImitationLearningModel(pl.LightningModule):
    def __init__(
        self,
        encoder_num_layer,
        encoder_emb_dim,
        encoder_load_path,
        encoder_optimize,
        decoder_num_layers,
        decoder_hidden_dim,
        decoder_code_dim,
        data_dir,
    ):
        super(ImitationLearningModel, self).__init__()
        self.encoder = GnnEncoder(num_layer=encoder_num_layer, emb_dim=encoder_emb_dim)
        self.encoder_optimize = encoder_optimize
        if encoder_load_path != "":
            self.encoder.load_state_dict(torch.load(encoder_load_path))

        if not self.encoder_optimize:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.sequence_handler = SequenceHandler(data_dir)
        num_vocabs = len(self.sequence_handler.vocabulary)
        self.decoder = RnnDecoder(
            num_layers=decoder_num_layers,
            input_dim=num_vocabs,
            output_dim=num_vocabs,
            hidden_dim=decoder_hidden_dim,
            code_dim=decoder_code_dim,
        )
        
        self.save_hyperparameters()

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("imitation")
        group.add_argument("--encoder_optimize", action="store_true")
        group.add_argument("--encoder_num_layer", type=int, default=5)
        group.add_argument("--encoder_emb_dim", type=int, default=300)
        group.add_argument("--encoder_load_path", type=str, default="")
        group.add_argument("--decoder_num_layers", type=int, default=3)
        group.add_argument("--decoder_hidden_dim", type=int, default=1024)
        group.add_argument("--decoder_code_dim", type=int, default=300)
        return parser

    def training_step(self, batched_data, batch_idx):
        batched_sequence_data, batched_pyg_data = batched_data
        with torch.no_grad():
            codes = self.encoder(batched_pyg_data)
            codes = torch.nn.functional.normalize(codes, p=2, dim=1)

        logits = self.decoder(batched_sequence_data, codes)
        loss = compute_sequence_cross_entropy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )
        elem_acc, sequence_acc = compute_sequence_accuracy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )

        self.log("train/loss/total", loss, on_step=True, logger=True)
        self.log("train/acc/element", elem_acc, on_step=True, logger=True)
        self.log("train/acc/sequence", sequence_acc, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        batched_sequence_data, batched_pyg_data = batched_data
        with torch.no_grad():
            codes = self.encoder(batched_pyg_data)
            logits = self.decoder(batched_sequence_data, codes)

        loss = compute_sequence_cross_entropy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )
        elem_acc, sequence_acc = compute_sequence_accuracy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )

        self.log("validation/loss/total", loss, logger=True)
        self.log("validation/acc/element", elem_acc, logger=True)
        self.log("validation/acc/sequence", sequence_acc, logger=True)

    def configure_optimizers(self):
        params = list(self.decoder.parameters())
        if self.encoder_optimize:
            params += list(self.encoder.parameters())

        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]
