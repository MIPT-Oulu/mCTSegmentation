import matplotlib
import matplotlib.pyplot as plt

import argparse
import glob
import os
import pickle
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_root', default='../workdir/snapshots')
    parser.add_argument('--snapshot_prefix', default='2019_0')
    parser.add_argument('--spacing', type=float, default=3.)
    parser.add_argument('--save_dir', default='../')
    parser.add_argument('--extension', choices=['pdf', 'png'], default='png')
    args = parser.parse_args()

    snapshots = glob.glob(os.path.join(args.snapshots_root, f'{args.snapshot_prefix}*'))
    experiments = {}
    for snp in snapshots:
        with open(os.path.join(snp, 'session.pkl'), 'rb') as f:
            session_backup = pickle.load(f)

        loss_type = session_backup['args'][0].loss
        loss_weight = session_backup['args'][0].loss_weight
        log_jaccard = getattr(session_backup['args'][0], "log_jaccard", False)
        for t in [0.3, 0.5, 0.6]:
            experiments[(loss_type, loss_weight, log_jaccard, t)] = pd.read_pickle(os.path.join(snp, 'oof_inference',
                                                                                   f'results_{t}.pkl'))
    os.makedirs(os.path.join(args.save_dir, 'pics'), exist_ok=True)

    for metric in ['IoU', 'Dice', 'VS']:
        matplotlib.rcParams.update({'font.size': 14})
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        axs.grid()
        axs.set_xlabel('Pad [$\mu$M]')
        axs.set_ylabel(metric)

        for setting_key, setting, color in [('BCE-$\log (IoU)$', ('combined', 0.5, True, 0.3), 'r'),
                                            ('BCE', ('bce', 0.5, False, 0.5), 'b'),
                                            ('Focal loss $\\alpha=0.25,\gamma=2$', ('focal', 0.5, False, 0.5), 'g')]:
            exp = experiments[setting]
            exp = exp[exp.metric == metric]
            val_columns = list(filter(lambda x: 'val@' in x, exp.columns.tolist()))
            pads = list(map(lambda x: int(x.split('@')[1]) * args.spacing, val_columns))

            mean = exp[val_columns].median(0).values
            plt.plot(pads, mean, color=color, label=setting_key)
            plt.plot(pads, mean, f'{color}o')

        axs.set_ylim(0.5, 1)
        plt.legend()
        plt.savefig(os.path.join(args.save_dir,
                                 'pics', f'{metric}.{args.extension}'), bbox_inches='tight')
        plt.close()


