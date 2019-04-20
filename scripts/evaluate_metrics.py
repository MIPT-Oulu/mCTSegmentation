import os
import argparse
import glob
from tqdm import tqdm
import pickle
import numpy as np
import h5py
import gc
from mctseg.imutils import read_stack


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--snapshots_root', default='')
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_backup = pickle.load(f)
