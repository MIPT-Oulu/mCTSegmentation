import cv2
import sys
import pickle
import argparse
import os
import glob
import numpy as np

from termcolor import colored
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm


from kvs import GlobalKVS
from mctseg.training.dataset import SegmentationDataset
from mctseg.imutils import read_gs_mask_ocv, read_gs_ocv
from mctseg.training import session

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS(None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--tta', type=bool, default=False)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_backup = pickle.load(f)

        args.model = session_backup['args'][0].model
        args.n_inputs = session_backup['args'][0].n_inputs
        args.n_classes = session_backup['args'][0].n_classes
        args.bw = session_backup['args'][0].bw
        args.depth = session_backup['args'][0].depth
        args.cdepth = session_backup['args'][0].cdepth
        args.seed = session_backup['args'][0].seed
        # For compatibility with the old arguments
        args.model = 'unet' if session_backup['args'][0].model == 'training' else session_backup['args'][0].model

    kvs.update('args', args)

    metadata = session_backup[f'metadata'][0]
    for sample_name, _ in metadata.groupby(by='sample_id'):
        os.makedirs(os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', sample_name), exist_ok=True)

    predicts = []
    fnames = []
    gt = []
    for fold_id, _, val_set in session_backup['cv_split'][0]:
        print(colored('====> ', 'green') + f'Loading fold [{fold_id}]')
        snapshot_name = glob.glob(os.path.join(args.snapshots_root, args.snapshot, f'fold_{fold_id}*.pth'))
        if len(snapshot_name) == 0:
            continue
        snapshot_name = snapshot_name[0]

        net = session.init_model(ignore_data_parallel=True)
        net.load_state_dict(torch.load(snapshot_name))

        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net).to('cuda')
        net.eval()

        if args.tta:
            raise NotImplementedError('TTA has not yet been implemented')
        else:
            test_trf = session_backup['val_trf'][0]

        val_dataset = SegmentationDataset(split=val_set,
                                          trf=session_backup['val_trf'][0],
                                          read_img=read_gs_ocv,
                                          read_mask=read_gs_mask_ocv)
        val_loader = DataLoader(val_dataset, batch_size=args.bs,
                                num_workers=args.n_threads,
                                sampler=SequentialSampler(val_dataset))

        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Predicting fold {fold_id}:'):
                img = batch['img']
                sample_ids = batch['sample_id']
                fnames = batch['fname']
                predicts = torch.sigmoid(net(img)).mul(255).to('cpu').numpy().astype(np.uint8)

                for idx, fname in enumerate(fnames):
                    pred_mask = predicts[idx].squeeze()
                    cv2.imwrite(os.path.join(args.snapshots_root,
                                             args.snapshot,
                                             'oof_inference', sample_ids[idx], fname), pred_mask)


