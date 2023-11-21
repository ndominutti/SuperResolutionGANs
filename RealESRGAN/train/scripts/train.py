"""
ATTENTION!

This file is not of my authorship; it is a copy from the excellent work created by user xinntao on GitHub.
I have copied this work to make minor modifications necessary for my implementation in SageMaker.

Original work by: xinntao (https://github.com/xinntao/Real-ESRGAN)
"""
import os.path as osp
from basicsr.train import train_pipeline

# import realesrgan.archs
# import realesrgan.data
# import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print('Root path: '+root_path)
    train_pipeline(root_path)
