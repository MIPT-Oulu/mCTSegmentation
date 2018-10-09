import torch
import solt.data as sld
import numpy as np
import cv2

def gs2tens(x):
    return torch.from_numpy(x.squeeze()).unsqueeze(0).long()


def img_mask2solt(img, mask):
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    return sld.DataContainer((img, mask.squeeze()), 'IM')


def solt2img_mask(dc: sld.DataContainer):
    if dc.data_format != 'IM':
        raise ValueError
    return dc.data[0], dc.data[1]


def read_gs_ocv(fname):
    return np.expand_dims(cv2.imread(fname, 0), -1)
