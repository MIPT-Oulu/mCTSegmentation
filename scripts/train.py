import sys
import os
import time
from termcolor import colored

import numpy as np
from tensorboardX import SummaryWriter

import cv2
import torch.optim.lr_scheduler as lr_scheduler

import mctseg.unet.session as session
import mctseg.unet.metrics as metrics
import mctseg.unet.dataset as dataset
import mctseg.unet.utils as utils
from mctseg.utils import GlobalKVS


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    dataset.init_metadata()
    session.init_data_processing()
    dataset.init_folds()

    for fold_id, X_train, X_val in kvs['cv_split']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)

        net = session.init_model()
        optimizer = session.init_optimizer(net)
        criterion = session.init_loss()

        scheduler = lr_scheduler.MultiStepLR(optimizer, kvs['args'].lr_drop)
        train_loader, val_loader = session.init_loaders(X_train, X_val)

        writer = SummaryWriter(os.path.join(kvs['args'].logs,
                                            'mCT_PTA_segmentation',
                                            'fold_{}'.format(fold_id), kvs['snapshot_name']))

        for epoch in range(kvs['args'].n_epochs):
            print(colored('==> ', 'green') + f'Training [{epoch}] with LR {scheduler.get_lr()}')
            kvs.update('cur_epoch', epoch)
            train_loss = utils.train_epoch(net, train_loader, optimizer, criterion)
            val_loss, conf_matrix = utils.validate_epoch(net, val_loader, criterion)

            metrics.log_metrics(writer, train_loss, val_loss, conf_matrix)
            utils.save_checkpoint(net, 'val_loss', 'lt')
            scheduler.step()




