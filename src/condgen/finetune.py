from tqdm import tqdm
import torch
from data.sequence.dataset import SequenceDataset
from data.score.dataset import ScoreDataset
from data.util import ZipDataset

def run_finetune(model, score_func_name, run):
    # seed=1
    model = model.cuda()
    #model.eval()
    ae = model.autoencoder
    device = model.device
    score_embedding = torch.nn.Linear(1, model.hparams.code_dim).cuda()
    
    optimizer = torch.optim.Adam(list(ae.parameters()) + list(score_embedding.parameters()), lr=1e-3)

    #
    input_dataset = SequenceDataset(model.hparams.data_dir, "train_labeled")
    score_dataset = ScoreDataset(model.hparams.data_dir, [score_func_name], "train_labeled")
    dataset = ZipDataset(input_dataset, score_dataset)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            collate_fn=dataset.collate,
            num_workers=8,
        )
    
    
    for epoch in tqdm(range(100)):
        for batched_input_data, batched_score_data in loader:
            batched_input_data = [tsr.cuda() for tsr in batched_input_data]
            batched_score_data = batched_score_data.cuda()
            codes = score_embedding(batched_score_data)
            decoder_out = ae.decoder(batched_input_data, codes)
            recon_loss, recon_statistics = ae.decoder.compute_recon_loss(decoder_out, batched_input_data)
            
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            for key, val in recon_statistics.items():
                run[f"{key}"].log(val)
