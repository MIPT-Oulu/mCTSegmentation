import cv2
import argparse
import os
import glob
import numpy as np
from tqdm import tqdm

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_ds_path', default='/media/lext/FAST/PTA_segmentation_project/Data/original')
    parser.add_argument('--pre_processed_ds_path',
                        default='/media/lext/FAST/PTA_segmentation_project/Data/pre_processed')
    parser.add_argument('--crop_size_xy', default=600)
    parser.add_argument('--crop_size_z', default=512)
    args = parser.parse_args()

    stacks = glob.glob(os.path.join(args.source_ds_path, '*'))
    for stack_path in stacks:
        if os.path.isdir(os.path.join(args.pre_processed_ds_path, stack_path.split('/')[-1])):
            continue

        imgs_names = glob.glob(os.path.join(stack_path, 'train', '*.png'))
        imgs_names = list(filter(lambda x: '_xz_' in x, imgs_names))
        imgs_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        mask_names = glob.glob(os.path.join(stack_path, 'trainMask', '*.png'))
        mask_names = list(filter(lambda x: '_xz_' in x, mask_names))
        mask_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        stack = None
        mask = None

        for i, fname in tqdm(enumerate(imgs_names), total=len(imgs_names),
                             desc='Reading Sample: ' + stack_path.split('/')[-1]):
            img = cv2.imread(fname, 0)

            if stack is None:
                stack = np.zeros((img.shape[0], len(imgs_names), img.shape[1]), dtype=np.uint8)
            stack[:, i, :] = img

        mask = np.zeros((img.shape[0], len(mask_names), img.shape[1]), dtype=np.uint8)
        for i, fname in tqdm(enumerate(mask_names), total=len(mask_names),
                             desc='Reading Mask: ' + stack_path.split('/')[-1]):
            img = cv2.imread(fname, 0)

            if img.shape[0] > mask.shape[0]:
                img = img[:mask.shape[0]]
            if img.shape[1] > mask.shape[2]:
                img = img[:, :mask.shape[2]]

            mask[:img.shape[0], i, :img.shape[1]] = img

        sample_mask_z = stack[:stack.shape[0] // 3].sum(0).astype(float)
        sample_mask_z -= sample_mask_z.min()
        sample_mask_z /= sample_mask_z.max()
        sample_mask_z = sample_mask_z > 0.1
        sample_mask_z = sample_mask_z.astype(np.uint8) * 255

        _, cnts, _ = cv2.findContours(sample_mask_z, 1, 2)  # Can be replaced by argmax for efficiency
        cnts.sort(key=cv2.contourArea)

        M = cv2.moments(cnts[-1])
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        stack = stack[:args.crop_size_z,
                (cy - args.crop_size_xy // 2):(cy + args.crop_size_xy // 2),
                (cx - args.crop_size_xy // 2):(cx + args.crop_size_xy // 2)]

        mask = mask[:args.crop_size_z,
               (cy - args.crop_size_xy // 2):
               (cy + args.crop_size_xy // 2),
               (cx - args.crop_size_xy // 2):(cx + args.crop_size_xy // 2)]

        os.makedirs(os.path.join(args.pre_processed_ds_path, stack_path.split('/')[-1], 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(args.pre_processed_ds_path, stack_path.split('/')[-1], 'masks'), exist_ok=True)

        # ZX
        for y_ind in tqdm(range(stack.shape[1]), total=stack.shape[1], desc='Saving [ZX] ' + stack_path.split('/')[-1]):
            slice_i = stack[:, y_ind, :].squeeze()
            mask_i = mask[:, y_ind, :].squeeze()

            cv2.imwrite(os.path.join(args.pre_processed_ds_path,
                                     stack_path.split('/')[-1], 'imgs', f'ZX_{y_ind}.png'), slice_i)

            cv2.imwrite(os.path.join(args.pre_processed_ds_path,
                                     stack_path.split('/')[-1], 'masks', f'ZX_{y_ind}.png'), mask_i)

        # ZY
        for x_ind in tqdm(range(stack.shape[2]), total=stack.shape[1], desc='Saving [ZY] ' + stack_path.split('/')[-1]):
            slice_i = stack[:, :, x_ind].squeeze()
            mask_i = mask[:, :, x_ind].squeeze()

            cv2.imwrite(os.path.join(args.pre_processed_ds_path,
                                     stack_path.split('/')[-1], 'imgs', f'ZY_{x_ind}.png'), slice_i)

            cv2.imwrite(os.path.join(args.pre_processed_ds_path,
                                     stack_path.split('/')[-1], 'masks', f'ZY_{x_ind}.png'), mask_i)
