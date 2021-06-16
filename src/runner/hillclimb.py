from collections import defaultdict

import torch
from tqdm import tqdm
from util.priority_queue import MaxRewardPriorityQueue
from scoring.factory import get_scoring_func
from vocabulary import seq2smiles, smiles2seq
from torch.nn.utils.rnn import pad_sequence

class HillClimber:
    def __init__(self, steps, warmup_steps, num_samplings_per_step, num_updates_per_step, sample_size, batch_size, queue_size):
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.num_samplings_per_step = num_samplings_per_step
        self.num_updates_per_step = num_updates_per_step
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.queue_size = queue_size

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("hillclimb")
        group.add_argument("--hillclimb_steps", type=int, default=200)
        group.add_argument("--hillclimb_warmup_steps", type=int, default=5)
        group.add_argument("--hillclimb_num_samplings_per_step", type=int, default=8)
        group.add_argument("--hillclimb_num_updates_per_step", type=int, default=8)
        group.add_argument("--hillclimb_sample_size", type=int, default=1024)
        group.add_argument("--hillclimb_batch_size", type=int, default=256)        
        group.add_argument("--hillclimb_queue_size", type=int, default=1024)
        return parser
    

    def run(self, dataset, model, optimizer, storage,logger):
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        for idx, scoring_func_name in [
            #(1, "celecoxib"),
            #(2, "troglitazone"),
            #(3, "thiothixene"),
            #(4, "aripiprazole"),
            #(5, "albuterol"),
            #(6, "mestranol"),
            #(7, "c11h24"),
            #(8, "c9h10n2o2pf2cl"),
            (9, "camphor_menthol"),
            (10, "tadalafil_sildenafil"),
            (11, "osimertinib"),
            (12, "fexofenadine"),
            (13, "ranolazine"),
            (14, "perindopril"),
            (15, "amlodipine"),
            (16, "sitagliptin"),
            (17, "zaleplon"),
            (18, "valsartan_smarts"),
            #(19, "decoration_hop"),
            #(20, "scaffold_hop"),
            #(21, "penalized_logp"),
        ]:
            scoring_func = get_scoring_func(scoring_func_name)
            storage.elems = []
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)

            warmup = True
            for step in tqdm(range(self.steps)):
                if step > self.warmup_steps:
                    warmup = False

                statistics = self.run_step(
                    dataset, 
                    model,
                    optimizer,
                    storage,
                    scoring_func,
                    warmup,
                )
                logger.log(statistics, prefix=f"{idx}_{scoring_func_name}")
    
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

                seqs = [smiles2seq(string, dataset.tokenizer, dataset.vocab) for string in strings]
                lengths = torch.tensor([seq.size(0) for seq in seqs])
                seqs = pad_sequence(seqs, batch_first=True, padding_value=0)

                seqs = seqs.cuda()
                lengths = lengths.cuda()
                loss, step_statistics = model.global_step(seqs, lengths)
                for key, val in step_statistics.items():
                    statistics[key] += val / self.num_updates_per_step

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        return statistics
