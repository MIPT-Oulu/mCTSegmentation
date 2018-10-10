import argparse
import torch
import numpy as np
import os
import time
from mctseg.utils import GlobalKVS, git_info


def init_session():
    # Getting the arguments
    args = parse_args()
    # Initializing the seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Creating teh snapshot
    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    os.makedirs(os.path.join(args.snapshots, snapshot_name), exist_ok=True)

    kvs = GlobalKVS()
    res = git_info()
    if res is not None:
        kvs.update('git branch name', res[0])
        kvs.update('git commit id', res[1])
    else:
        kvs.update('git branch name', None)
        kvs.update('git commit id', None)

    kvs.update('pytorch_version', torch.__version__)

    if torch.cuda.is_available():
        kvs.update('cuda', torch.version.cuda)
        kvs.update('gpus', None)
    else:
        kvs.update('cuda', None)
        kvs.update('gpus', torch.cuda.device_count())

    kvs.update('snapshot_name', snapshot_name)
    kvs.update('args', args)
    kvs.update('train_loss', None, dict)
    kvs.update('val_loss', None, dict)
    kvs.update('val_metrics', None, dict)
    kvs.save_pkl(os.path.join(args.snapshots, snapshot_name, 'session.pkl'))

    return args, snapshot_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/media/lext/FAST/PTA_segmentation_project/Data/pre_processed')
    parser.add_argument('--snapshots', default='/media/lext/FAST/PTA_segmentation_project/snapshots/')
    parser.add_argument('--logs', default='/media/lext/FAST/PTA_segmentation_project/logs/')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--val_bs', type=int, default=32)
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--start_val', type=int, default=-1)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--cd', type=int, default=1)
    parser.add_argument('--bw', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    return args
