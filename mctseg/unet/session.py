import argparse
import numpy as np
import time
from mctseg.utils import GlobalKVS, git_info


import torch
from termcolor import colored
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise EnvironmentError('The code must be run on GPU.')


def init_session():
    kvs = GlobalKVS()

    # Getting the arguments
    args = parse_args()
    # Initializing the seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Creating the snapshot
    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    os.makedirs(os.path.join(args.snapshots, snapshot_name), exist_ok=True)

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
        kvs.update('gpus', torch.cuda.device_count())
    else:
        kvs.update('cuda', None)
        kvs.update('gpus', None)

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
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--n_inputs', type=int, default=1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--start_val', type=int, default=-1)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--cdepth', type=int, default=1)
    parser.add_argument('--bw', type=int, default=24)
    parser.add_argument('--crop_x', type=int, default=256)
    parser.add_argument('--crop_y', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    return args


def save_checkpoint(cur_snapshot_name, model, loss_value, prev_model, best_loss):
    if prev_model is None:
        print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
        torch.save(model.state_dict(), cur_snapshot_name)
        prev_model = cur_snapshot_name
        best_loss = loss_value
        return prev_model, best_loss
    else:
        if loss_value < best_loss:
            print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
            os.remove(prev_model)
            best_loss = loss_value
            torch.save(model.state_dict(), cur_snapshot_name)
            prev_model = cur_snapshot_name
    return prev_model, best_loss