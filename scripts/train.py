import sys
import os
from termcolor import colored
from tensorboardX import SummaryWriter

import cv2
import torch.optim.lr_scheduler as lr_scheduler

import mctseg.training.session as session
import mctseg.training.dataset as dataset
import mctseg.training.utils as utils
from kvs import GlobalKVS


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    session.init_session()
    dataset.init_metadata()
    session.init_data_processing()
    dataset.init_folds()

    kvs = GlobalKVS()
    for fold_id, X_train, X_val in kvs['cv_split']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)

        net = session.init_model()
        optimizer = session.init_optimizer(net)
        criterion = session.init_loss()

        scheduler = lr_scheduler.MultiStepLR(optimizer, kvs['args'].lr_drop)
        train_loader, val_loader = session.init_loaders(X_train, X_val)

        writer = SummaryWriter(os.path.join(kvs['args'].workdir, 'snapshots', kvs['snapshot_name'],
                                            'logs', 'fold_{}'.format(fold_id), kvs['snapshot_name']))

        for epoch in range(kvs['args'].n_epochs):
            print(colored('==> ', 'green') + f'Training epoch [{epoch}] with LR {scheduler.get_lr()}')
            kvs.update('cur_epoch', epoch)
            train_loss, _ = utils.pass_epoch(net, train_loader, optimizer, criterion)
            val_loss, conf_matrix = utils.pass_epoch(net, val_loader, None, criterion)

            utils.log_metrics(writer, train_loss, val_loss, conf_matrix)
            utils.save_checkpoint(net, 'val_loss', 'lt')
            scheduler.step()




