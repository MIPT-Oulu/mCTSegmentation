import torch.utils.data as data


class SegmentationDataset(data.Dataset):
    def __init__(self, split, trf, read_img, read_mask):
        self.split = split
        self.transforms = trf
        self.read_img = read_img
        self.read_mask = read_mask

    def __getitem__(self, idx):
        entry = self.split.iloc[idx]
        img_fname = entry.slice_fname
        mask_fname = entry.mask_fname

        img = self.read_img(img_fname)
        mask = self.read_mask(mask_fname)

        img, mask = self.transforms((img, mask))

        return {'img': img, 'mask': mask}

    def __len__(self):
        return self.split.shape[0]