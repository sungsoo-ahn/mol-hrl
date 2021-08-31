import torch
from torch.functional import Tensor
from pl_module.conddecoder import CondDecoderModule
from module.plug.vae import PlugVariationalAutoEncoder
from jtnn import *

from data.util import load_tokenizer
from data.graph.dataset import GraphDataset
from data.sequence.dataset import SequenceDataset
from data.util import ZipDataset, load_score_list, load_smiles_list, TensorDataset
from data.score.dataset import ScoreDataset, load_statistics
from data.score.score import load_scorer
from torch.utils.data import DataLoader

from tqdm import tqdm

class WrappedJTNNVAE(JTNNVAE):
    def encode_latent_mean(self, smiles_list):
        mol_batch = []
        encode_success = []
        for s in tqdm(smiles_list):
            mol_tree = MolTree(s)
            mol_tree.recover()

            try:
                for node in mol_tree.nodes:
                    self.vocab.get_index(node.smiles)
                mol_batch.append(mol_tree)
                encode_success.append(True)
            except:
                encode_success.append(False)                   
            
        _, tree_vec, mol_vec = self.encode(mol_batch)
        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(mol_vec)

        return torch.cat([tree_mean,mol_mean], dim=1), torch.tensor(encode_success)

    def decode_latent_mean(self, codes, prob_decode=False):
        tree_vec, mol_vec = torch.chunk(codes, chunks=2, dim=1)
        return self.decode(tree_vec, mol_vec, prob_decode=prob_decode)
        
class JTNNModule(CondDecoderModule):
    def setup_layers(self, hparams):
        vocab = [x.strip("\r\n ") for x in open(hparams.vocab_path)] 
        vocab = Vocab(vocab)
    
        self.jtnn = WrappedJTNNVAE(vocab, 450, 56, 3, stereo=True)
        self.plug_vae = PlugVariationalAutoEncoder(
            num_layers=hparams.plug_num_layers, 
            code_dim=hparams.code_dim, 
            hidden_dim=hparams.plug_hidden_dim, 
            latent_dim=hparams.plug_latent_dim,
            )

        state_dict = torch.load(hparams.load_checkpoint_path)
        missing = {k: v for k, v in self.jtnn.state_dict().items() if k not in state_dict}
        state_dict.update(missing) 
        self.jtnn.load_state_dict(state_dict)
    
    def setup_datasets(self, hparams):
        pass
    
    def postsetup_datasets(self):
        self.scorer = load_scorer(self.hparams.task)        
        self.score_mean, self.score_std, self.score_success_margin = load_statistics(self.hparams.task)

        self.jtnn.cuda()
        self.jtnn.eval()
        
        def build_dataset(task, split):
            smiles_list = load_smiles_list(task, split)
            score_list = load_score_list(task, split)
            
            latent_tsrs, encode_success = self.jtnn.encode_latent_mean(smiles_list)
            score_tsrs = torch.tensor(score_list)[encode_success]

            code_dataset = TensorDataset(latent_tsrs.detach().cpu())
            score_dataset = TensorDataset(score_tsrs)

            return ZipDataset(code_dataset, score_dataset)
    
        self.train_dataset = build_dataset(self.hparams.task, "train")
        self.val_dataset = build_dataset(self.hparams.task, "valid")
        
        def collate(data_list):
            code_list, score_list = zip(*data_list)
            return torch.stack(code_list), torch.stack(score_list)
        
        self.collate = collate    

    @staticmethod
    def add_args(parser):
        
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-4)
    
        # Common - data
        parser.add_argument("--task", type=str, default="plogp")
        parser.add_argument("--batch_size", type=int, default=256)
        
        # model        
        parser.add_argument("--vocab_path", type=str, default="../resource/checkpoint/jtnn/vocab.txt")
        parser.add_argument("--load_checkpoint_path", type=str, default="../resource/checkpoint/jtnn/model.iter-4")

        # model - code
        parser.add_argument("--code_dim", type=int, default=56)
        
        # model - encoder
        parser.add_argument("--encoder_num_layers", type=int, default=5)
        parser.add_argument("--encoder_hidden_dim", type=int, default=256)

        # model - decoder
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        # model - plug lstm
        parser.add_argument("--plug_num_layers", type=int, default=2)
        parser.add_argument("--plug_hidden_dim", type=int, default=256)
        parser.add_argument("--plug_latent_dim", type=int, default=128)
        parser.add_argument("--plug_beta", type=float, default=1e-1)
        
        # sampling
        parser.add_argument("--num_queries", type=int, default=1000)
        parser.add_argument("--query_batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--max_len", type=int, default=120)

        return parser

    def shared_step(self, batched_data):
        batched_input_data, batched_cond_data = batched_data
        batched_cond_data = (batched_cond_data - self.score_mean) / self.score_std

        out, z, p, q = self.plug_vae(batched_input_data, batched_cond_data)
        
        recon_loss = torch.nn.functional.mse_loss(out, batched_input_data, reduction="mean")
        kl_loss = (q.log_prob(z) - p.log_prob(z)).mean()
        loss = recon_loss + self.hparams.plug_beta * kl_loss
        
        statistics = dict()        
        statistics["loss/total"] = loss
        statistics["loss/plug_recon"] = recon_loss
        statistics["loss/plug_kl"] = kl_loss

        return loss, statistics

    def decode_many_smiles(self, query, num_samples, max_len):
        batched_cond_data = torch.full((1, 1), query, device=self.device)
        batched_cond_data = (batched_cond_data - self.score_mean) / self.score_std
        smiles_list = []
        for _ in tqdm(range(num_samples)):
            with torch.no_grad():
                codes = self.plug_vae.decode(batched_cond_data)
                smiles = self.jtnn.decode_latent_mean(codes, prob_decode=False)
                smiles_list.append(smiles)

        return smiles_list
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.plug_vae.parameters(), lr=self.hparams.lr)
        return [optimizer]