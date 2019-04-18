import torch
import solt.data as sld
import cv2
import numpy as np


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


def read_stack(slices):
    stack = None

    for i, fname in enumerate(slices):
        img = cv2.imread(fname, 0)

        if stack is None:
            stack = np.zeros((img.shape[0], len(slices), img.shape[1]), dtype=np.float16)
        stack[:, i, :] = img

    stack -= stack.min()
    stack /= stack.max()

    return stack