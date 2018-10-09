import argparse


def parse_args():
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
    return args