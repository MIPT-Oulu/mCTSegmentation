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
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_backup = pickle.load(f)

    samples = os.listdir(os.path.join(args.snapshots_root, args.snapshot, 'oof_inference'))
    samples = list(filter(lambda x: 'hdf' not in x, samples))
    samples = list(filter(lambda x: 'pkl' not in x, samples))
    os.makedirs(os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', 'hdf'), exist_ok=True)
    for sample_id in tqdm(samples, total=len(samples)):
        slices_ZX = glob.glob(os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', sample_id, 'ZX_*.png'))
        slices_ZY = glob.glob(os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', sample_id, 'ZY_*.png'))

        slices_ZX.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        slices_ZY.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

        stack_ZX: np.ndarray = read_stack(slices_ZX)
        stack_ZY: np.ndarray = read_stack(slices_ZY)

        stack_combined = (stack_ZY.swapaxes(1, 2) + stack_ZX) / 2.

        sample_save = os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', 'hdf', f'{sample_id}.hdf5')
        h5 = h5py.File(sample_save, 'w')
        h5.create_dataset('ZX_data', data=stack_combined, dtype=stack_combined.dtype, compression='gzip')
        h5.close()
        gc.collect()
