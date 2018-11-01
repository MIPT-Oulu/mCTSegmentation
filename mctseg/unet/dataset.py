import torch.utils.data as data
import torch
import solt.data as sld
import cv2
import copy

from mctseg.utils import GlobalKVS
import glob
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os
import numpy as np
from sklearn.model_selection import GroupKFold


from functools import partial
import solt.transforms as slt
import solt.core as slc
from torchvision import transforms


class SegmentationDataset(data.Dataset):
    def __init__(self, split, trf, read_img, read_mask):
        self.split = split
        self.transforms = trf
        self.read_img = read_img
        self.read_mask = read_mask

    def __getitem__(self, idx):
        entry = self.split.iloc[idx]
        img_fname = entry.img_fname
        mask_fname = entry.mask_fname

        img = self.read_img(img_fname)
        mask = self.read_mask(mask_fname)

        img, mask = self.transforms((img, mask))

        return {'img': img, 'mask': mask}

    def __len__(self):
        return self.split.shape[0]


def gs2tens(x, dtype='f'):
    if dtype == 'f':
        return torch.from_numpy(x.squeeze()).unsqueeze(0).float()
    elif dtype == 'l':
        return torch.from_numpy(x.squeeze()).unsqueeze(0).long()
    else:
        raise NotImplementedError


def img_mask2solt(imgmask):
    img, mask = imgmask
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    return sld.DataContainer((img, mask.squeeze()), 'IM')


def solt2img_mask(dc: sld.DataContainer):
    if dc.data_format != 'IM':
        raise ValueError
    return dc.data[0], dc.data[1]


def read_gs_ocv(fname):
    return np.expand_dims(cv2.imread(fname, 0), -1)


def read_gs_mask_ocv(fname):
    return np.expand_dims((cv2.imread(fname, 0) > 0).astype(np.float32), -1)


def apply_by_index(items, transform, idx=0):
    """Applies callable to certain objects in iterable using given indices.
    Parameters
    ----------
    items: tuple or list
    transform: callable
    idx: int or tuple or or list None
    Returns
    -------
    result: tuple
    """
    if idx is None:
        return items
    if not isinstance(items, (tuple, list)):
        raise TypeError
    if not isinstance(idx, (int, tuple, list)):
        raise TypeError

    if isinstance(idx, int):
        idx = [idx, ]

    idx = set(idx)
    res = []
    for i, item in enumerate(items):
        if i in idx:
            res.append(transform(item))
        else:
            res.append(copy.deepcopy(item))

    return tuple(res)


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

    grades = pd.read_csv(kvs['args'].grades)

    n_subj = np.unique(metadata.subject_id.values).shape[0]
    if n_subj <= kvs['args'].train_size:
        raise ValueError

    metadata = pd.merge(metadata, grades, on='sample_id')
    gss = GroupShuffleSplit(n_splits=1,
                            train_size=kvs['args'].train_size,
                            test_size=n_subj-kvs['args'].train_size,
                            random_state=kvs['args'].seed)

    train_ind, test_ind = next(gss.split(metadata, y=metadata.grade, groups=metadata.subject_id))

    metadata_train = metadata.iloc[train_ind]
    metadata_test = metadata.iloc[test_ind]

    kvs.update('metadata_train', metadata_train)
    kvs.update('metadata_test', metadata_test)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))

    return metadata


def init_folds():
    kvs = GlobalKVS()
    gkf = GroupKFold(kvs['args'].n_folds)
    cv_split = []
    for fold_id, (train_ind, val_ind) in enumerate(gkf.split(X=kvs['metadata_train'],
                                                             y=kvs['metadata_train'].grade,
                                                             groups=kvs['metadata_train'].subject_id)):

        if kvs['args'].fold != -1 and fold_id != kvs['args'].fold:
            continue

        cv_split.append((fold_id,
                         kvs['metadata_train'].iloc[train_ind],
                         kvs['metadata_train'].iloc[val_ind]))

        kvs.update(f'losses_fold_[{fold_id}]', None, list)
        kvs.update(f'val_metrics_fold_[{fold_id}]', None, list)

    kvs.update('cv_split', cv_split)


def init_train_augmentation_pipeline():
    kvs = GlobalKVS()
    ppl = transforms.Compose([
        img_mask2solt,
        slc.Stream([
            slt.RandomFlip(axis=1, p=0.5),
            slt.ImageGammaCorrection(gamma_range=(0.5, 2), p=0.5),
            slt.PadTransform(pad_to=(kvs['args'].crop_x+1, kvs['args'].crop_y+1)),
            slt.CropTransform(crop_size=(kvs['args'].crop_x, kvs['args'].crop_y), crop_mode='r')
        ]),
        solt2img_mask,
        partial(apply_by_index, transform=gs2tens, idx=[0, 1]),
    ])
    return ppl
