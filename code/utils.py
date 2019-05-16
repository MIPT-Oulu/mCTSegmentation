import argparse
import glob
import pandas as pd
import os
from deeppipeline.kvs import GlobalKVS
import torch.optim.lr_scheduler as lr_scheduler


def gen_image_id(fname,  sample_id):
    prj, slice_num = fname.split('/')[-1].split('.')[0].split('_')
    return f'{sample_id}_{slice_num}_{prj}'


def init_metadata():
    kvs = GlobalKVS()

    imgs = glob.glob(os.path.join(kvs['args'].dataset, '*', 'imgs', '*.png'))
    imgs.sort(key=lambda x: x.split('/')[-1])

    masks = glob.glob(os.path.join(kvs['args'].dataset, '*', 'masks', '*.png'))
    masks.sort(key=lambda x: x.split('/')[-1])

    sample_id = list(map(lambda x: x.split('/')[-3], imgs))
    subject_id = list(map(lambda x: x.split('/')[-3].split('_')[0], imgs))

    metadata = pd.DataFrame(data={'img_fname': imgs, 'mask_fname': masks,
                                  'sample_id': sample_id, 'subject_id': subject_id})

    metadata['sample_subject_proj'] = metadata.apply(lambda x: gen_image_id(x.img_fname, x.sample_id), 1)
    grades = pd.read_csv(kvs['args'].grades)
    metadata = pd.merge(metadata, grades, on='sample_id')
    kvs.update('metadata', metadata)
    return metadata


def init_scheduler(optimizer):
    kvs = GlobalKVS()
    return lr_scheduler.MultiStepLR(optimizer, kvs['args'].lr_drop)


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../Data/pre_processed')
    parser.add_argument('--grades', default='../Data/grades.csv')
    parser.add_argument('--workdir', default='../workdir')
    parser.add_argument('--model', type=str, choices=['unet'], default='unet')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--loss', choices=['bce', 'jaccard', 'combined', 'focal'], default='bce')
    parser.add_argument('--loss_weight', type=float, default=0.5)
    parser.add_argument('--log_jaccard', type=bool, default=False)
    parser.add_argument('--binary_threshold', type=float, default=0.5)
    parser.add_argument('--val_bs', type=int, default=32)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--skip_train', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_inputs', type=int, default=1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--gamma_min', type=float, default=0.5)
    parser.add_argument('--gamma_max', type=float, default=2)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--n_threads', type=int, default=20)
    parser.add_argument('--start_val', type=int, default=-1)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--cdepth', type=int, default=1)
    parser.add_argument('--bw', type=int, default=24)
    parser.add_argument('--crop_x', type=int, default=384)
    parser.add_argument('--crop_y', type=int, default=640)
    parser.add_argument('--pad_x', type=int, default=800)
    parser.add_argument('--pad_y', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_drop', nargs='+', default=[20, 25, 28], type=int)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    return args
