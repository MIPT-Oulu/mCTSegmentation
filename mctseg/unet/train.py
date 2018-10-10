import sys
import os
import gc
import time
import argparse
import pickle
import random
import glob
import pandas as pd
from termcolor import colored

import numpy as np
from sklearn.model_selection import GroupKFold
from tensorboardX import SummaryWriter

import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from mctseg.unet.session import init_session
from mctseg.unet.metadata import init_metadata
from mctseg.unet.model import UNet
from mctseg.unet.dataset import init_augmentations, SegmentationDataset
from mctseg.unet.loss import BCEWithLogitsLoss2d, BinaryDiceLoss, CombinedLoss
from mctseg.utils import GlobalKVS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    init_session()
    init_metadata()
    init_augmentations()

    gkf = GroupKFold(kvs['args'].n_folds)
    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(kvs['metadata'], groups=kvs['metadata'].subject_id)):
        print(train_idx.shape)







