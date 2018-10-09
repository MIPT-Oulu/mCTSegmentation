import torch
import solt.data as sld
import numpy as np
import cv2
from termcolor import colored
import os


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


def save_checkpoint(cur_snapshot_name, model, loss_value, prev_model, best_loss):
    if prev_model is None:
        print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
        torch.save(model.state_dict(), cur_snapshot_name)
        prev_model = cur_snapshot_name
        best_loss = loss_value
        return prev_model, best_loss
    else:
        if loss_value < best_loss:
            print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
            os.remove(prev_model)
            best_loss = loss_value
            torch.save(model.state_dict(), cur_snapshot_name)
            prev_model = cur_snapshot_name
    return prev_model, best_loss
