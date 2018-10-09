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

from mctseg.unet.init_train import init_train
from mctseg.unet.model import UNet
from mctseg.unet.dataset import SegmentationDataset
from mctseg.unet.loss import BCEWithLogitsLoss2d, BinaryDiceLoss, CombinedLoss
from mctseg.utils import read_gs_ocv, GlobalLogger, git_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    args, snapshot_name = init_train()