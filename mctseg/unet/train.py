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
import torch.optim.lr_scheduler as lr_scheduler

from mctseg.unet.session import init_session
from mctseg.unet.metadata import init_metadata
from mctseg.unet.model import UNet
from mctseg.unet.dataset import init_data_processing, SegmentationDataset, read_gs_ocv, read_gs_mask_ocv
import mctseg.unet.utils as utils
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

        criterion_dict = {'combined': CombinedLoss([BCEWithLogitsLoss2d(), BinaryDiceLoss()]),
                          'bce': BCEWithLogitsLoss2d(),
                          'dice': BinaryDiceLoss(),
                          'wbce': BCEWithLogitsLoss2d(weight=torch.from_numpy(kvs['class_weights']))}

        X_train = kvs['metadata'].iloc[train_idx]
        X_val = kvs['metadata'].iloc[train_idx]
        train_dataset = SegmentationDataset(split=X_train,
                                            trf=kvs['train_trf'],
                                            read_img=read_gs_ocv,
                                            read_mask=read_gs_mask_ocv)

        val_dataset = SegmentationDataset(split=X_val,
                                          trf=kvs['val_trf'],
                                          read_img=read_gs_ocv,
                                          read_mask=read_gs_mask_ocv)

        train_loader = DataLoader(train_dataset, batch_size=kvs['args'].bs,
                                  num_workers=kvs['args'].n_threads, shuffle=True,
                                  drop_last=True,
                                  worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))

        val_loader = DataLoader(val_dataset, batch_size=kvs['args'].bs,
                                  num_workers=kvs['args'].n_threads)

        prev_model = None
        best_loss = 10e10
        for epoch in range(kvs['args'].n_epochs):
            start = time.time()
            print(colored('==> ', 'green') + f'Training [{epoch}] with LR {scheduler.get_lr()}')
            train_loss = utils.train_epoch(epoch, net, optimizer, train_loader, criterion, kvs['args'].n_epochs)
            val_loss, conf_matrix = utils.validate_epoch(net, val_loader, criterion)
            epoch_time = np.round(time.time() - start, 4)
            dices = {'dice_{}'.format(cls): dice for cls, dice in enumerate(utils.calculate_dice(conf_matrix))}

            print(colored('==> ', 'green') + 'Metrics:')
            print(colored('====> ', 'green') + 'Train loss:', train_loss)
            print(colored('====> ', 'green') + 'Val loss:', val_loss)
            print(colored('====> ', 'green') + f'Val AUC: {val_auc}')
            scheduler.step()







