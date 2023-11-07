"""
This module is a helper to create the train dataset using albumentations library.
We will apply:
* for both HR and LR images: 
    1) random crops of 96px*96px
    2) flip horizontally randomly (with a probability of .5)
    3) rotate 90º randomly (with a probability of .5)

* for HR images:
    1) normalization with mu=.5 and sigma=.5

* for LR images:
    1) resizing to a 24px*24px image using BICUBIC interpolation (this is the step that
    reduces the image quality)
    2) normalization with mu=0 and sigma=1 -> to get the [-1,1] authors used in the paper
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=24, height=24, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=96, height=96),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
