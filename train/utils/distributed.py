import random
import torch
import numpy as np
import builtins as __builtin__
from datetime import timedelta

def init_distributed():

    torch.distributed.init_process_group(
        backend='nccl' if torch.distributed.is_nccl_available() else 'gloo',
        timeout=timedelta(seconds=10800),
        init_method="env://"
    )

    rank = torch.distributed.get_rank()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    torch.distributed.barrier()

    set_seed(42, rank)
    enable_print(rank==0)

    print(f'Init distributed mode. {device}')


def set_seed(seed, device):
    # fix the seed for reproducibility
    seed = seed + int(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_print(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
