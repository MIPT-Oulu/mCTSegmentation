import sys
import os
import time
from termcolor import colored

import numpy as np
from sklearn.model_selection import GroupKFold
from tensorboardX import SummaryWriter

import cv2
import torch.optim.lr_scheduler as lr_scheduler

import mctseg.unet.session as session
import mctseg.unet.metrics as metrics
import mctseg.unet.metadata as metadata
import mctseg.unet.utils as utils
from mctseg.utils import GlobalKVS


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    metadata.init_metadata()
    session.init_data_processing()

    gkf = GroupKFold(kvs['args'].n_folds)
    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(kvs['metadata'], groups=kvs['metadata'].subject_id)):
        net = session.init_model()
        optimizer = session.init_optimizer(net)

        scheduler = lr_scheduler.MultiStepLR(optimizer, kvs['args'].lr_drop)
        criterion = session.init_loss()

        X_train = kvs['metadata'].iloc[train_idx]
        X_val = kvs['metadata'].iloc[train_idx]
        train_loader, val_loader = session.init_loaders(X_train, X_val)

        prev_model = None
        best_loss = 10e10

        for epoch in range(kvs['args'].n_epochs):
            start = time.time()
            print(colored('==> ', 'green') + f'Training [{epoch}] with LR {scheduler.get_lr()}')
            train_loss = utils.train_epoch(fold_id, epoch, net,
                                           optimizer, train_loader, criterion, kvs['args'].n_epochs)

            val_loss, conf_matrix = utils.validate_epoch(epoch, kvs['args'].n_epochs, net,
                                                         val_loader, criterion, kvs['args'].n_classes)

            epoch_time = np.round(time.time() - start, 4)
            dices = {'dice_{}'.format(cls): dice for cls, dice in enumerate(metrics.calculate_dice(conf_matrix))}

            print(colored('==> ', 'green') + 'Metrics:')
            print(colored('====> ', 'green') + 'Train loss:', train_loss)
            print(colored('====> ', 'green') + 'Val loss:', val_loss)
            print(colored('====> ', 'green') + f'Val Dices: {dices}')

            scheduler.step()
            cur_snapshot_name = os.path.join(kvs['args'].snapshots, kvs['snapshot_name'],
                                             f'fold_{fold_id}_epoch_{epoch+1}.pth')

            prev_model, best_loss = utils.save_checkpoint(cur_snapshot_name, net.module,
                                                          val_loss, prev_model, best_loss)






