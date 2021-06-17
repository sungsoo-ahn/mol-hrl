import argparse
from runner.pretrain import Pretrainer
from runner.hillclimb import HillClimber
from model.generator import FeatureBasedGenerator
from dataset import FeaturizedSmilesDataset
from util.priority_queue import MaxRewardPriorityQueue
from logger import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Pretrainer.add_args(parser)
    HillClimber.add_args(parser)
    FeatureBasedGenerator.add_args(parser)
    FeaturizedSmilesDataset.add_args(parser)
    Logger.add_args(parser)
    args = parser.parse_args()

    dataset = FeaturizedSmilesDataset(
        args.data_dir,
        args.data_tag,
        args.data_aug_randomize_smiles,
        args.data_aug_mutate,
    )
    model = FeatureBasedGenerator(
        args.generator_hidden_dim,
        args.generator_num_layers,
        dataset.vocab,
    ).cuda()
    pretrainer = Pretrainer(
        args.pretrain_epochs,
        args.pretrain_batch_size,
        args.pretrain_dir,
        args.pretrain_tag,
    )
    hillclimber = HillClimber(
        args.hillclimb_steps,
        args.hillclimb_warmup_steps,
        args.hillclimb_num_samplings_per_step,
        args.hillclimb_num_updates_per_step,
        args.hillclimb_sample_size,
        args.hillclimb_batch_size,
        args.hillclimb_queue_size,
    )
    storage = MaxRewardPriorityQueue()
    logger = Logger(args.logger_use_neptune)

    #
    logger.set("parameters", vars(args))

    pretrain_optimizer = model.get_pretrain_optimizer()
    pretrainer.run(dataset, model, pretrain_optimizer, logger)

    #
    hillclimb_optimizer = model.get_hillclimb_optimizer()
    hillclimber.run(dataset, model, hillclimb_optimizer, storage, logger)
