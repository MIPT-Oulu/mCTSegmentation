import os
import argparse
import glob
from tqdm import tqdm
import pickle
import numpy as np
import h5py
import gc
from mctseg.imutils import read_stack
from mctseg.evaluation.metrics import calculate_confusion_matrix_from_arrays, calculate_iou, calculate_dice
from mctseg.evaluation.metrics import calculate_volumetric_similarity
import pandas as pd


def make_surf_vol(stack, surf_pad=5):
    mask_vol = np.zeros(stack.shape)

    # Searching for the upper boundary upside down
    surf = np.flip(stack, 0).argmax(0)
    # Creating a surface in 3d
    for x in range(surf.shape[1]):
        for y in range(surf.shape[0]):
            z1 = (surf[y, x]-surf_pad) * (surf[y, x] >= surf_pad)
            z2 = (surf[y, x]+surf_pad)
            mask_vol[z1:z2, y, x] = 1.0
    mask_vol[:mask_vol.shape[0]//3] = 0
    mask_vol = np.flip(mask_vol, 0).copy()
    upper_bound = surf.flatten().max() + surf_pad
    return mask_vol[:upper_bound]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--pred_threshold', type=float, default=0.5)
    parser.add_argument('--pad_min', type=int, default=5)
    parser.add_argument('--pad_max', type=int, default=50)
    parser.add_argument('--n_pads', type=int, default=10)

    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_backup = pickle.load(f)
    predicts_dir = os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', 'hdf')
    predicts_fnames = os.listdir(predicts_dir)

    pbar = tqdm(total=len(predicts_fnames))
    results = []
    paddings = np.linspace(args.pad_min, args.pad_max, args.n_pads, dtype=int)
    for predict_fname in predicts_fnames:
        sample_id = predict_fname.split('.hdf5')[0]
        pbar.set_description(f'Processing sample [{sample_id}]:')
        h5pred = h5py.File(os.path.join(predicts_dir, predict_fname), 'r')
        pred_stack = h5pred['ZX_data'][:] > args.pred_threshold

        masks = glob.glob(os.path.join(args.dataset_dir, sample_id, 'masks', 'ZX*.png'))
        masks.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
        gt_stack = read_stack(masks) > 0.9

        iou_scores = []
        dice_scores = []
        vs_scores = []
        for pad in paddings:
            pbar.set_description(f'Processing sample [{sample_id} / pad {pad}]:')
            mask_zone = make_surf_vol(gt_stack, surf_pad=pad)
            masked_preds = pred_stack[:mask_zone.shape[0]] * mask_zone
            masked_gt = gt_stack[:mask_zone.shape[0]] * mask_zone

            conf_mat = calculate_confusion_matrix_from_arrays(masked_preds,
                                                              masked_gt, 2)
            # IoU
            iou = calculate_iou(conf_mat)[1]
            iou_scores.append(iou)
            # Dice
            dice = calculate_dice(conf_mat)[1]
            dice_scores.append(dice)
            # VD
            volumetric_sim = calculate_volumetric_similarity(conf_mat)[1]
            vs_scores.append(volumetric_sim)

        results.append([sample_id, 'IoU', ] + iou_scores)
        results.append([sample_id, 'Dice', ] + dice_scores)
        results.append([sample_id, 'VS', ] + vs_scores)

        gc.collect()
        pbar.update()
    pbar.close()

    results_df = pd.DataFrame(data=results, columns=['sample', 'metric', ]+[f'val@{pad_val}' for pad_val in paddings])
    results_df.to_pickle(os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', 'results.pkl'))


