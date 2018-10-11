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
        partial(apply_by_index, transform=gs2tens, idx=[0, 1]),
    ])
    return ppl
