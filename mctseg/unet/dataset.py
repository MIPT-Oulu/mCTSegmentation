import torch.utils.data as data
import torch
import solt.data as sld
import solt.transforms as slt
import solt.core as slc
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from mctseg.utils import GlobalKVS


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


def init_data_processing():
    kvs = GlobalKVS()
    train_ppl = init_train_augmentation_pipeline()

    dataset = SegmentationDataset(split=kvs['metadata'],
                                  trf=train_ppl,
                                  read_img=read_gs_ocv,
                                  read_mask=read_gs_ocv)

    mean_vector, std_vector, class_weights = init_mean_std(snapshots_dir=kvs['args'].snapshots,
                                                           dataset=dataset,
                                                           batch_size=kvs['args'].bs,
                                                           n_threads=kvs['args'].n_threads,
                                                           n_classes=kvs['args'].n_classes+1)


def init_train_augmentation_pipeline():
    kvs = GlobalKVS()
    ppl = transforms.Compose([
        lambda x: img_mask2solt(*x),
        slc.Stream([
            slt.RandomFlip(axis=1, p=0.5),
            slc.SelectiveStream([
                slc.Stream([
                    slt.RandomRotate(rotation_range=(-10, 10), p=1),
                    slt.RandomShear(range_x=(-0.1, 0.1), p=0.5),
                    slt.RandomShear(range_y=(-0.1, 0.1), p=0.5),
                    slt.RandomScale(range_x=(0.9, 1.2), same=True, p=1),
                    slt.ImageGammaCorrection(gamma_range=(0.8, 1.3))
                ]),
                slc.Stream(),
                slt.ImageGammaCorrection(gamma_range=(0.8, 1.3)),
            ]),
            slt.PadTransform(pad_to=(kvs['args'].crop_x+1, kvs['args'].crop_y+1)),
            slt.CropTransform(crop_size=(kvs['args'].crop_x, kvs['args'].crop_y), crop_mode='r')
        ]),
        solt2img_mask,
        lambda x: (gs2tens(x[0]), gs2tens(x[1]))

    ])
    return ppl


def gs2tens(x, dtype='f'):
    if dtype == 'f':
        return torch.from_numpy(x.squeeze()).unsqueeze(0).float()
    elif dtype == 'l':
        return torch.from_numpy(x.squeeze()).unsqueeze(0).long()
    else:
        raise NotImplementedError


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
        for batch in tqdm(tmp_loader, total=len(tmp_loader)):
            imgs = batch['img']
            masks = batch['mask']
            if mean_vector is None:
                mean_vector = np.zeros(imgs.size(1))
                std_vector = np.zeros(imgs.size(1))
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += imgs[:, j, :, :].mean()
                std_vector[j] += imgs[:, j, :, :].std()

            for j in range(class_weights.shape[0]):
                class_weights[j] += np.sum(masks.numpy() == j)
            num_pixels += np.prod(masks.size())

        mean_vector /= len(tmp_loader)
        std_vector /= len(tmp_loader)
        class_weights /= num_pixels
        class_weights = 1 / class_weights
        class_weights /= class_weights.max()
        np.save(os.path.join(snapshots_dir, 'mean_std_weights.npy'), [mean_vector.astype(np.float32),
                                                                      std_vector.astype(np.float32),
                                                                      class_weights.astype(np.float32)])

    return mean_vector, std_vector, class_weights
