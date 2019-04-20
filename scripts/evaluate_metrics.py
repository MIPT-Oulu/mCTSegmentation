import os
import argparse
import glob
from tqdm import tqdm
import pickle
import numpy as np
import h5py
import gc
from mctseg.imutils import read_stack
from mctseg.evaluation.metrics import calculate_confusion_matrix_from_arrays, calculate_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_backup = pickle.load(f)
    predicts_dir = os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', 'hdf')
    predicts_fnames = os.listdir(predicts_dir)

    pbar = tqdm(total=len(predicts_fnames))
    iou_scores = []
    for predict_fname in predicts_fnames:
        sample_id = predict_fname.split('.hdf5')[0]
        pbar.set_description(f'Processing sample [{sample_id}]:')
        h5pred = h5py.File(os.path.join(predicts_dir, predict_fname), 'r')
        pred_stack = h5pred['ZX_data'][:]

        masks = glob.glob(os.path.join(args.dataset_dir, sample_id, 'masks', 'ZX*.png'))
        masks.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
        gt_stack = read_stack(masks)

        conf_matr = calculate_confusion_matrix_from_arrays(pred_stack > 0.5, gt_stack > 0.5, 2)
        iou = calculate_iou(conf_matr)
        iou_scores.append(iou)

        pbar.update()
    pbar.close()



