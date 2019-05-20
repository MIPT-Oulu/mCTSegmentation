import cv2
import pickle
import argparse
import os


from deeppipeline.kvs import GlobalKVS
from deeppipeline.io import read_gs_binary_mask_ocv, read_gs_ocv
from deeppipeline.segmentation.evaluation import run_oof_binary

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


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

    kvs.update('args', args)
    run_oof_binary(args=args, session_backup=session_backup, read_img=read_gs_ocv,
                   read_mask=read_gs_binary_mask_ocv, img_group_id_colname='sample_id')
