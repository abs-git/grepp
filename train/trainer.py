import argparse
from itertools import product

import torch
import yaml
from easydict import EasyDict as edict

from grepp.train.utils.distributed import init_distributed
from grepp.train.utils.engine import Train


def main(args):

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = edict(cfg)

    if args.checkpoint != None:
        cfg.default.checkpoint=args.checkpoint

    if args.output_dir != None:
        cfg.output.root=args.output_dir

    learning_rates = [0.001]
    batch_sizes = [4, 8, 16]
    perspectives = [0.3]
    optimizers = ['adam', 'sgd']
    schedulers = ['steplr', 'exponentiallr']

    for lr, batch, pers, optim, sche in product(learning_rates,
                                                batch_sizes,
                                                perspectives,
                                                optimizers,
                                                schedulers):

        cfg.default.update({'lr':lr})
        cfg.default.update({'batch_size':batch})
        cfg.default.update({'perspective':pers})
        cfg.default.update({'optimizer':optim})
        cfg.default.update({'scheduler':sche})

        if torch.cuda.is_available():
            init_distributed()
            trainer = Train(cfg)
            trainer()
        else:
            trainer = Train(cfg)
            trainer()


def parsing_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parsing_args()
    main(args)
