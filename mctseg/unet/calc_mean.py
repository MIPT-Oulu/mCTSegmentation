import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def calc_mean(snapshots_dir, dataset, batch_size, n_threads, n_classes):
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
