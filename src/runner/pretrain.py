from collections import defaultdict

import torch
from tqdm import tqdm


class Pretrainer:
    def __init__(self, epochs, batch_size, dir, tag):
        self.epochs = epochs
        self.batch_size = batch_size
        self.dir = dir
        self.tag = tag

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("pretrain")
        group.add_argument("--pretrain_epochs", type=int, default=100)
        group.add_argument("--pretrain_batch_size", type=int, default=256)
        group.add_argument("--pretrain_dir", type=str, default="../resource/checkpoint")
        group.add_argument("--pretrain_tag", type=str, default="pretrain")
        return parser

    def run(self, dataset, model, optimizer, logger):
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=8,
        )

        for epoch in range(self.epochs):
            statistics = self.run_epoch(model, optimizer, loader)
            logger.log(statistics)

            perf = statistics["loss/sum"]
            if epoch == 0 or perf > best_perf:
                best_perf = perf
                state_dict = model.state_dict()
                torch.save(state_dict, f"{self.dir}/{self.tag}.pth")

    def run_epoch(self, model, optimizer, loader):
        statistics = defaultdict(float)
        for batched_data in tqdm(loader):
            loss, step_statistics = model.pretrain_step(batched_data)
            for key, val in step_statistics.items():
                statistics[key] += val / len(loader)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        return statistics
