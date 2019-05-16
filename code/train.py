import cv2

from deeppipeline.segmentation.training.core import train_n_folds
from deeppipeline.io import read_gs_ocv, read_gs_binary_mask_ocv

from .mctseg.utils import parse_train_args, init_metadata, init_scheduler

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    train_n_folds(init_args=parse_train_args, init_metadata=init_metadata, init_scheduler=init_scheduler,
                  img_reader=read_gs_ocv, mask_reader=read_gs_binary_mask_ocv,
                  img_group_id_colname='subject_id', img_id_colname='sample_subject_proj', img_class_colname='grade')


