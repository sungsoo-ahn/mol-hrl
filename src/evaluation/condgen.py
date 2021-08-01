from tqdm import tqdm
import torch
from module.pl_autoencoder import AutoEncoderModule
from data.sequence.dataset import SequenceDataset
from data.score.dataset import ScoreDataset
from data.util import ZipDataset
from data.score.factory import get_scoring_func


def run_finetune(checkpoint_path, score_func_name, run):
    # seed=1
    EVAL_BATCH_SIZE = 256
    LR = 1e-3

    model = AutoEncoderModule.load_from_checkpoint(checkpoint_path)
    model = model.cuda()
    ae = model.autoencoder
    device = model.device
    score_embedding = torch.nn.Linear(1, model.hparams.code_dim).cuda()
    optimizer = torch.optim.Adam(list(ae.parameters()) + list(score_embedding.parameters()), lr=LR)

    if score_func_name == "penalized_logp":
        eval_scores = [0.0, 2.0, 4.0]
    elif score_func_name == "logp":
        eval_scores = [1.5, 3.0, 4.5]
    elif score_func_name == "qed":
        eval_scores = [0.5, 0.7, 0.9]
    elif score_func_name == "molwt":
        eval_scores = [250, 350, 450]

    #
    input_dataset = SequenceDataset(model.hparams.data_dir, "train_labeled")
    score_dataset = ScoreDataset(model.hparams.data_dir, [score_func_name], "train_labeled")
    dataset = ZipDataset(input_dataset, score_dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, collate_fn=dataset.collate, num_workers=8,
    )

    _, smiles_score_func, corrupt_score = get_scoring_func(score_func_name)
    invalid_scores = score_dataset.raw_tsrs.min()

    def score_codes(codes):
        smiles_list = ae.decoder.sample_smiles(codes.to(device), argmax=True)
        scores = smiles_score_func(smiles_list)
        scores = torch.FloatTensor(scores)
        scores[scores < corrupt_score + 1e-6] = invalid_scores
        return scores.unsqueeze(1)

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

        for target_score in eval_scores:
            batched_score_data = torch.full((EVAL_BATCH_SIZE, 1), target_score).cuda()
            codes = score_embedding(batched_score_data)
            scores = score_codes(codes)
            loss = torch.nn.functional.mse_loss(target_score, scores)

            run[f"{score_func_name}/{target_score}/mse"].log(loss)
