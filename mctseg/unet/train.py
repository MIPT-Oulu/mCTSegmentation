import sys
import os
import gc
import time
import argparse
import pickle
import random
import glob
import pandas as pd
from termcolor import colored

import numpy as np
from sklearn.model_selection import GroupKFold
from tensorboardX import SummaryWriter

import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from mctseg.unet.session import init_session
from mctseg.unet.metadata import init_metadata
from mctseg.unet.model import UNet
from mctseg.unet.dataset import init_data_processing, SegmentationDataset, read_gs_ocv
from mctseg.unet.loss import BCEWithLogitsLoss2d, BinaryDiceLoss, CombinedLoss
from mctseg.utils import GlobalKVS


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    init_session()
    init_metadata()
    init_data_processing()

    gkf = GroupKFold(kvs['args'].n_folds)
    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(kvs['metadata'], groups=kvs['metadata'].subject_id)):
        net = UNet(bw=kvs['args'].bw, depth=kvs['args'].depth,
                   center_depth=kvs['args'].cdepth,
                   n_inputs=kvs['args'].n_inputs,
                   n_classes=kvs['args'].n_classes,
                   activation='relu')
        if kvs['gpus'] > 1:
            net = nn.DataParallel(net).to('cuda')

        net = net.to('cuda')

        X_train = kvs['metadata'].iloc[train_idx]
        X_val = kvs['metadata'].iloc[train_idx]
        train_dataset = SegmentationDataset(split=X_train,
                                            trf=kvs['train_trf'],
                                            read_img=read_gs_ocv,
                                            read_mask=read_gs_ocv)

        val_dataset = SegmentationDataset(split=X_val,
                                            trf=kvs['val_trf'],
                                            read_img=read_gs_ocv,
                                            read_mask=read_gs_ocv)









