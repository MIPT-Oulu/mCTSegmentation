import torch.utils.data as data
import torch
import solt.data as sld
import solt.transforms as slt
import solt.core as slc
import cv2
import os
import numpy as np

import copy



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

