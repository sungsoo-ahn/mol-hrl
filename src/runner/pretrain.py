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

    def run(self, train_dataset, vali_dataset, model, optimizer, logger):
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=8,
        )
        vali_loader = torch.utils.data.DataLoader(
            dataset=vali_dataset,
            batch_size=self.batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=8,
        )

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
