from tqdm import tqdm
import torch
from data.util import ZipDataset, TensorDataset
from module.pl_autoencoder import AutoEncoderModule

class DecoupledAutoEncoderModule(AutoEncoderModule):
    def __init__(self, hparams):
        super(DecoupledAutoEncoderModule, self).__init__(hparams)
        if self.hparams.stage == 1:
            for param in self.autoencoder.encoder.parameters():
                param.requires_grad=False
            
            self.autoencoder.encoder.eval()
            
    @staticmethod
    def add_args(parser):
        super(DecoupledAutoEncoderModule, DecoupledAutoEncoderModule).add_args(parser)
        parser.add_argument("--stage", type=int, default=0)

    def setup_datasets(self):
        if self.hparams.stage == 0:
            self.train_dataset = self.autoencoder.get_input_dataset("train")
            self.val_dataset = self.autoencoder.get_input_dataset("val")
        
        elif self.hparams.stage == 1:
            train_target_dataset = self.autoencoder.get_target_dataset("train")
            val_target_dataset = self.autoencoder.get_target_dataset("val")
            train_code_dataset = self.extract_code_dataset("train")
            val_code_dataset = self.extract_code_dataset("val")
            self.train_dataset = ZipDataset(train_target_dataset, train_code_dataset)
            self.val_dataset = ZipDataset(val_target_dataset, val_code_dataset)

    def extract_code_dataset(self, split):
        self.autoencoder.encoder.eval()
        self.autoencoder.encoder.cuda()
        
        dataset = self.autoencoder.encoder.get_dataset(split)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )
        codes = []
        print(f"Extracting codes from {split} dataset...")
        for batched_data in tqdm(dataloader):
            batched_data = batched_data.cuda()
            with torch.no_grad():
                batched_codes = self.autoencoder.encoder(batched_data)
            
            codes.append(batched_codes.detach().cpu())
        
        codes = torch.cat(codes, dim=0)
        return TensorDataset(codes)

    def shared_step(self, batched_data):
        if self.hparams.stage == 0:
            _, loss, statistics = self.autoencoder.update_encoder_loss(batched_data)
        elif self.hparams.stage == 1:
            batched_target_data, codes = batched_data
            loss, statistics = self.autoencoder.update_decoder_loss(batched_target_data, codes)
        
        return loss, statistics
    
    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log(f"stage{self.hparams.stage}/train/loss/total", loss, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"stage{self.hparams.stage}/train/{key}", val, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log(f"stage{self.hparams.stage}/validation/loss/total", loss, on_step=False, logger=True)
        for key, val in statistics.items():
            self.log(f"stage{self.hparams.stage}/validation/{key}", val, on_step=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.hparams.lr)

        return [optimizer]