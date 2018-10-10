from mctseg.utils import GlobalKVS
import glob
import os
import pandas as pd


def init_metadata():
    kvs = GlobalKVS()

    imgs = glob.glob(os.path.join(kvs['args'].dataset, '*', 'imgs', '*.png'))
    imgs.sort(key=lambda x: x.split('/')[-1])

    masks = glob.glob(os.path.join(kvs['args'].dataset, '*', 'masks', '*.png'))
    masks.sort(key=lambda x: x.split('/')[-1])

    sample_id = list(map(lambda x: x.split('/')[-3], imgs))
    subject_id = list(map(lambda x: x.split('/')[-3].split('_')[0], imgs))

    metadata = pd.DataFrame(data={'img_fname': imgs, 'mask_fname': masks,
                                  'sample_id': sample_id, 'subject_id': subject_id})

    kvs.update('metadata', metadata)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))

    return metadata