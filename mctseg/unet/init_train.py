import argparse
import torch
import numpy as np
import os
import time
from mctseg.utils import GlobalLogger, git_info


def init_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/media/lext/FAST/PTA_segmentation_project/Data/post_processed')
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    os.makedirs(os.path.join(args.snapshots, snapshot_name), exist_ok=True)

    logger = GlobalLogger()
    res = git_info()
    if res is not None:
        logger.update('git branch name', res[0])
        logger.update('git commit id', res[1])
    logger.update('pytorch_version', torch.__version__)
    if torch.cuda.is_available():
        logger.update('cuda', torch.version.cuda)

    else:
        logger.update('cuda', None)
    logger.update('gpus', torch.cuda.device_count())
    logger.update('snapshot_name', snapshot_name)
    logger.update('args', vars(args))
    for fold_id in range(args.n_folds):
        logger.update(f'[{fold_id}] train_loss', None, list)
        logger.update(f'[{fold_id}] val_loss', None, list)
        logger.update(f'[{fold_id}] val_dice', None, list)
    logger.save(os.path.join(args.snapshots, snapshot_name, 'log.json'))
    
    return args, snapshot_name
