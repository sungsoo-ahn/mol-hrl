import argparse
from runner.pretrain import Pretrainer
from runner.hillclimb import HillClimber
from model.generator import PenalizedLogPFeatureBasedGenerator
from dataset import PenalizedLogPFeaturizedSmilesDataset
from util.priority_queue import MaxRewardPriorityQueue
from logger import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Pretrainer.add_args(parser)
    HillClimber.add_args(parser)
    PenalizedLogPFeatureBasedGenerator.add_args(parser)
    PenalizedLogPFeaturizedSmilesDataset.add_args(parser)
    Logger.add_args(parser)
    args = parser.parse_args()

    train_dataset = PenalizedLogPFeaturizedSmilesDataset(
        args.data_dir,
        args.data_tag,
        args.data_aug_randomize_smiles,
        args.data_aug_mutate,
    )
    vali_dataset = PenalizedLogPFeaturizedSmilesDataset(
        args.data_dir,
        "zinc_small",
        args.data_aug_randomize_smiles,
        args.data_aug_mutate,
    )

    model = PenalizedLogPFeatureBasedGenerator(
        args.generator_hidden_dim,
        args.generator_num_layers,
        train_dataset.vocab,
        train_dataset.tokenizer,
    ).cuda()
    pretrainer = Pretrainer(
        args.pretrain_epochs,
        args.pretrain_batch_size,
        args.pretrain_dir,
        args.pretrain_tag,
    )
    #hillclimber = HillClimber(
    #    args.hillclimb_steps,
    #    args.hillclimb_warmup_steps,
    #    args.hillclimb_num_samplings_per_step,
    #    args.hillclimb_num_updates_per_step,
    #    args.hillclimb_sample_size,
    #    args.hillclimb_batch_size,
    #    args.hillclimb_queue_size,
    #)
    #storage = MaxRewardPriorityQueue()
    logger = Logger(args.logger_use_neptune)

    #
    logger.set("parameters", vars(args))

    pretrain_optimizer = model.get_optimizer()
    pretrainer.run(train_dataset, vali_dataset, model, pretrain_optimizer, logger)

    #
    #hillclimb_optimizer = model.get_hillclimb_optimizer()
    #hillclimber.run(dataset, model, hillclimb_optimizer, storage, logger)
