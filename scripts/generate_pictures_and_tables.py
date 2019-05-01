import matplotlib
import matplotlib.pyplot as plt

import argparse
import glob
import os
import pickle
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot_prefix', default='2019_04_')
    parser.add_argument('--spacing', type=float, default=3.)
    args = parser.parse_args()

    snapshots = glob.glob(os.path.join(args.snapshots_root, f'{args.snapshot_prefix}*'))
    experiments = {}
    for snp in snapshots:
        with open(os.path.join(snp, 'session.pkl'), 'rb') as f:
            session_backup = pickle.load(f)

        loss_type = session_backup['args'][0].loss
        loss_weight = session_backup['args'][0].loss_weight
        log_jaccard = getattr(session_backup['args'][0], "log_jaccard", False)
        experiments[(loss_type, loss_weight, log_jaccard)] = pd.read_pickle(os.path.join(snp, 'oof_inference',
                                                                                         'results.pkl'))

    for metric in ['IoU', 'Dice', 'VS']:
        matplotlib.rcParams.update({'font.size': 14})
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        axs.grid()
        axs.set_xlabel('Pad [$\mu$M]')
        axs.set_ylabel(metric)

        for setting_key, setting, color in [('Jaccard', ('jaccard', 0.5, True), 'r'),
                                            ('BCE', ('bce', 0.5, False), 'b'),
                                            ('Jaccard', ('jaccard', 0.5, False), 'g')]:
            exp = experiments[setting]
            exp = exp[exp.metric == metric]
            val_columns = list(filter(lambda x: 'val@' in x, exp.columns.tolist()))
            pads = list(map(lambda x: int(x.split('@')[1]) * args.spacing, val_columns))

            mean = exp[val_columns].mean(0).values
            std = exp[val_columns].std(0).values
            plt.errorbar(pads, mean, yerr=std, fmt='o-', color=color, capsize=3)

        if metric == 'IoU':
            axs.set_ylim(0.3, 1)
        else:
            axs.set_ylim(0.6, 1)
        plt.show()


