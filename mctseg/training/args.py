import argparse


def parse_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/media/lext/FAST/PTA_segmentation_project/Data/pre_processed')
    parser.add_argument('--grades', default='/media/lext/FAST/PTA_segmentation_project/Data/grades.csv')
    parser.add_argument('--workdir', default='/media/lext/FAST/PTA_segmentation_project/workdir')
    parser.add_argument('--logs', default='/media/lext/FAST/PTA_segmentation_project/logs/')
    parser.add_argument('--model', type=str, choices=['unet'], default='unet')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--loss', choices=['bce', 'jaccard', 'combined'], default='bce')
    parser.add_argument('--loss_weight', type=float, default=0.5)
    parser.add_argument('--val_bs', type=int, default=32)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--skip_train', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_inputs', type=int, default=1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--n_threads', type=int, default=20)
    parser.add_argument('--start_val', type=int, default=-1)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--cdepth', type=int, default=1)
    parser.add_argument('--bw', type=int, default=24)
    parser.add_argument('--crop_x', type=int, default=384)
    parser.add_argument('--crop_y', type=int, default=640)
    parser.add_argument('--pad_x', type=int, default=800)
    parser.add_argument('--pad_y', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_drop', nargs='+', default=[20, 25, 28])
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.lr_drop = list(map(int, args.lr_drop))

    return args

