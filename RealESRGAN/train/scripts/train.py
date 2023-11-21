# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

# import realesrgan.archs
# import realesrgan.data
# import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print('Root path: '+root_path)
    train_pipeline(root_path)
