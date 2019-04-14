import numpy as np
from termcolor import colored
from mctseg.utils import GlobalKVS
import os


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    """
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint64)
    return confusion_matrix


def calculate_dice(confusion_matrix):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    """

    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def log_metrics(writer, train_loss, val_loss, conf_matrix):
    kvs = GlobalKVS()

    dices = {'dice_{}'.format(cls): dice for cls, dice in enumerate(calculate_dice(conf_matrix))}
    ious = {'iou_{}'.format(cls): iou for cls, iou in enumerate(calculate_iou(conf_matrix))}
    print(colored('==> ', 'green') + 'Metrics:')
    print(colored('====> ', 'green') + 'Train loss:', train_loss)
    print(colored('====> ', 'green') + 'Val loss:', val_loss)
    print(colored('====> ', 'green') + f'Val Dice: {dices}')
    print(colored('====> ', 'green') + f'Val IoU: {ious}')
    dices_tb = {}
    for cls in range(1, len(dices)):
        dices_tb[f"Dice [{cls}]"] = dices[f"dice_{cls}"]

    ious_tb = {}
    for cls in range(1, len(ious)):
        ious_tb[f"IoU [{cls}]"] = ious[f"iou_{cls}"]

    to_log = {'train_loss': train_loss, 'val_loss': val_loss}
    # Tensorboard logging
    writer.add_scalars(f"Losses_{kvs['args'].model}", to_log, kvs['cur_epoch'])
    writer.add_scalars('Metrics', dices_tb, kvs['cur_epoch'])
    writer.add_scalars('Metrics', ious_tb, kvs['cur_epoch'])
    # KVS logging
    to_log.update({'epoch': kvs['cur_epoch']})
    val_metrics = {'epoch': kvs['cur_epoch']}
    val_metrics.update(to_log)
    val_metrics.update(dices)
    val_metrics.update({'conf_matrix': conf_matrix})

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', to_log)
    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', val_metrics)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
