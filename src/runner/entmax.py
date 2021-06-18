from collections import defaultdict

import torch
from tqdm import tqdm

from util.storage import ReplayBuffer

class Pretrainer:
    def __init__(self, buffer_size, epochs, steps_per_epoch, batch_size, sample_size, std_scale, dir, tag):
        self.buffer = ReplayBuffer(buffer_size)

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.std_scale = std_scale
        self.dir = dir
        self.tag = tag

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("pretrain")
        group.add_argument("--entmax_epochs", type=int, default=100)
        group.add_argument("--entmax_warmup_epochs", type=int, default=100)
        group.add_argument("--entmax_steps_per_epoch", type=int, default=1)
        group.add_argument("--entmax_buffer_size", type=int, default=50000)
        group.add_argument("--entmax_batch_size", type=int, default=256)
        group.add_argument("--entmax_sample_size", type=int, default=256)
        group.add_argument("--entmax_std_scale", type=float, default=1.5)
        group.add_argument("--entmax_dir", type=str, default="../resource/checkpoint")
        group.add_argument("--entmax_tag", type=str, default="entmax")
        return parser

    def run(self, model, optimizer, logger):
        for epoch in range(self.warmup_epochs):
            self.run_


        for epoch in range(self.epochs):
            statistics = self.run_epoch(model, optimizer, train_loader, vali_loader)
            logger.log(statistics, prefix="pretrain")

            perf = statistics["loss"]
            if epoch == 0 or perf < best_perf:
                best_perf = perf
                print(best_perf)
                state_dict = model.state_dict()
                torch.save(state_dict, f"{self.dir}/{self.tag}.pth")

        model.load_state_dict(state_dict)
        
    def run_epoch(self, model, optimizer, train_loader, vali_loader):
        statistics = defaultdict(float)
        for batched_data in tqdm(train_loader):
            loss, step_statistics = model.step(batched_data)
            for key, val in step_statistics.items():
                statistics[key] += val / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for batched_data in tqdm(vali_loader):
            with torch.no_grad():
                eval_statistics = model.eval(batched_data)
            
            for key, val in eval_statistics.items():
                statistics[key] += val / len(vali_loader)


        return statistics

    def run_step(self, dataset, model, optimizer, storage, scoring_func, warmup):
        for _ in range(self.num_samplings_per_step):
            with torch.no_grad():
                seqs, lengths, _ = model.sample(self.sample_size)

            strings = seq2smiles(seqs, dataset.tokenizer, dataset.vocab)
            scores = scoring_func(strings)
            seqs = seqs.cpu().split(1, dim=0)
            lengths = lengths.cpu().split(1, dim=0)

            storage.add_list(smis=strings, seqs=seqs, lengths=lengths, scores=scores)
            storage.squeeze_by_kth(k=self.queue_size)

        strings, seqs, lengths, scores = storage.get_elems()
        scores = torch.tensor(scores)

        statistics = defaultdict(float)
        statistics["score/max"] = scores.max().item()
        statistics["score/mean"] = scores.mean().item()
        statistics["score/guacamol"] = (
            torch.topk(scores, k=100)[0].mean().item()
            + torch.topk(scores, k=10)[0].mean().item()
            + scores.max().item()
        ) / 3

        if not warmup:
            for _ in range(self.num_updates_per_step):
                strings, seqs, lengths, scores = storage.sample_batch(self.batch_size)

                seqs = [
                    smiles2seq(string, dataset.tokenizer, dataset.vocab)
                    for string in strings
                ]
                lengths = torch.tensor([seq.size(0) for seq in seqs])
                seqs = pad_sequence(seqs, batch_first=True, padding_value=0)

                batched_data = (seqs, lengths)
                loss, step_statistics = model.hillclimb_step(batched_data)
                for key, val in step_statistics.items():
                    statistics[key] += val / self.num_updates_per_step

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        return statistics

