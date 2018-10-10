import torch.utils.data as data
import solt.data as sld
import torch
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


class SegmentationDataset(data.Dataset):
    def __init__(self, split, trf, read_img, read_mask):
        self.split = split
        self.transforms = trf
        self.read_img = read_img
        self.read_mask = read_mask

    def __getitem__(self, idx):
        entry = self.split.iloc[idx]
        img_fname = entry.img_fname
        mask_fname = entry.mask_fname

        img = self.read_img(img_fname)
        mask = self.read_mask(mask_fname)

        img, mask = self.transforms((img, mask))

        return {'img': img, 'mask': mask}

    def __len__(self):
        return self.split.shape[0]


def init_augmentations():
    pass


def gs2tens(x):
    return torch.from_numpy(x.squeeze()).unsqueeze(0).long()


def img_mask2solt(img, mask):
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    return sld.DataContainer((img, mask.squeeze()), 'IM')


def solt2img_mask(dc: sld.DataContainer):
    if dc.data_format != 'IM':
        raise ValueError
    return dc.data[0], dc.data[1]


def read_gs_ocv(fname):
    return np.expand_dims(cv2.imread(fname, 0), -1)


def init_mean_std(snapshots_dir, dataset, batch_size, n_threads, n_classes):
    if os.path.isfile(os.path.join(snapshots_dir, 'mean_std_weights.npy')):
        tmp = np.load(os.path.join(snapshots_dir, 'mean_std_weights.npy'))
        mean_vector, std_vector, class_weights = tmp
    else:
        tmp_loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_threads)
        mean_vector = None
        std_vector = None
        num_pixels = 0
        class_weights = np.zeros(n_classes)
        print('==> Calculating mean and std')
        for batch, mask in tqdm(tmp_loader, total=len(tmp_loader)):
            if mean_vector is None:
                mean_vector = np.zeros(batch.size(1))
                std_vector = np.zeros(batch.size(1))
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += batch[:, j, :, :].mean()
                std_vector[j] += batch[:, j, :, :].std()

            for j in range(class_weights.shape[0]):
                class_weights[j] += np.sum(mask.numpy() == j)
            num_pixels += np.prod(mask.size())

        mean_vector /= len(tmp_loader)
        std_vector /= len(tmp_loader)
        class_weights /= num_pixels
        class_weights = 1 / class_weights
        class_weights /= class_weights.max()
        np.save(os.path.join(snapshots_dir, 'mean_std_weights.npy'), [mean_vector.astype(np.float32),
                                                                      std_vector.astype(np.float32),
                                                                      class_weights.astype(np.float32)])

    return mean_vector, std_vector, class_weights
