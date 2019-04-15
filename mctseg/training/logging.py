from termcolor import colored
from mctseg.kvs import GlobalKVS
import os
from mctseg.evaluation.metrics import calculate_dice, calculate_iou


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
    writer.add_scalars('Metrics/Dice', dices_tb, kvs['cur_epoch'])
    writer.add_scalars('Metrics/IoU', ious_tb, kvs['cur_epoch'])
    # KVS logging
    to_log.update({'epoch': kvs['cur_epoch']})
    val_metrics = {'epoch': kvs['cur_epoch']}
    val_metrics.update(to_log)
    val_metrics.update(dices)
    val_metrics.update({'conf_matrix': conf_matrix})

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', to_log)
    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', val_metrics)

    kvs.save_pkl(os.path.join(kvs['args'].workdir, 'snapshots', kvs['snapshot_name'], 'session.pkl'))
