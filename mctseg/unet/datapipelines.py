from functools import partial
import solt.transforms as slt
import solt.core as slc
from torchvision import transforms
from mctseg.unet.dataset import img_mask2solt, solt2img_mask, apply_by_index, gs2tens
from mctseg.utils import GlobalKVS


def build_train_augmentation_pipeline():
    kvs = GlobalKVS()
    ppl = transforms.Compose([
        img_mask2solt,
        slc.Stream([
            slt.RandomFlip(axis=1, p=0.5),
            slt.ImageGammaCorrection(gamma_range=(0.5, 2), p=0.5),
            slt.PadTransform(pad_to=(kvs['args'].crop_x+1, kvs['args'].crop_y+1)),
            slt.CropTransform(crop_size=(kvs['args'].crop_x, kvs['args'].crop_y), crop_mode='r')
        ]),
        solt2img_mask,
        partial(apply_by_index, transform=gs2tens, idx=[0, 1]),
    ])
    return ppl
