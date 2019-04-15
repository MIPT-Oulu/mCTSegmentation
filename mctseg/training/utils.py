import torch
from tqdm import tqdm
import gc
import numpy as np
from termcolor import colored
import mctseg.evaluation.metrics as metrics
import os
from mctseg.kvs import GlobalKVS
import operator


def train_epoch(net, train_loader, optimizer, criterion):
    kvs = GlobalKVS()
    net.train(True)

    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    max_ep = kvs['args'].n_epochs

    running_loss = 0.0
    n_batches = len(train_loader)
    device = next(net.parameters()).device
    pbar = tqdm(total=n_batches, ncols=200)
    for i, entry in enumerate(train_loader):
        inputs = entry['img'].to(device)
        mask = entry['mask'].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f"Fold [{fold_id}] [{epoch} | {max_ep}] | "
                             f"Running loss {running_loss / (i + 1):.5f} / {loss.item():.5f}")
        pbar.update()

        gc.collect()
    gc.collect()
    pbar.close()

    return running_loss / n_batches


def validate_epoch(net, val_loader, criterion):
    kvs = GlobalKVS()
    net.train(False)

    epoch = kvs['cur_epoch']
    max_epoch = kvs['args'].n_epochs
    n_classes = kvs['args'].n_classes

    device = next(net.parameters()).device
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.uint64)
    val_loss = 0
    with torch.no_grad():
        for entry in tqdm(val_loader, total=len(val_loader), desc=f"[{epoch} / {max_epoch}] Val: "):
            img = entry['img'].to(device)
            mask = entry['mask'].to(device).squeeze()
            preds = net(img)
            val_loss += criterion(preds, mask).item()

            mask = mask.to('cpu').numpy()
            if n_classes == 2:
                preds = (preds.to('cpu').numpy() > 0.5).astype(float)
            elif n_classes > 2:
                preds = preds.to('cpu').numpy().argmax(axis=1)
            else:
                raise ValueError

            confusion_matrix += metrics.calculate_confusion_matrix_from_arrays(preds, mask, n_classes)

    val_loss /= len(val_loader)

    return val_loss, confusion_matrix


def save_checkpoint(net, val_metric_name, comparator='lt'):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    kvs = GlobalKVS()
    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    val_metric = kvs[f'val_metrics_fold_[{fold_id}]'][-1][0][val_metric_name]
    comparator = getattr(operator, comparator)
    cur_snapshot_name = os.path.join(os.path.join(kvs['args'].workdir, 'snapshots', kvs['snapshot_name'],
                                     f'fold_{fold_id}_epoch_{epoch}.pth'))

    if kvs['prev_model'] is None:
        print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
        torch.save(net.state_dict(), cur_snapshot_name)
        kvs.update('prev_model', cur_snapshot_name)
        kvs.update('best_val_metric', val_metric)

    else:
        if comparator(val_metric, kvs['best_val_metric']):
            print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
            os.remove(kvs['prev_model'])
            torch.save(net.state_dict(), cur_snapshot_name)
            kvs.update('prev_model', cur_snapshot_name)
            kvs.update('best_val_metric', val_metric)

    kvs.save_pkl(os.path.join(kvs['args'].workdir, 'snapshots', kvs['snapshot_name'], 'session.pkl'))
