from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import auxiliars
import numpy as np
import os
import argparse


# def download_dataset(args):
#     """
#     Load datasets from Hugging Face and save them to disk
    
#     Args:
#       data_path(str): HF path to the XNLI dataset
#     """
#     data = load_dataset("satellite-image-deep-learning/SODA-A", split='train[:1000]')
    
#     data.shuffle(seed=1)
#     data[:900].save_to_disk(args.save_path+'/train')
#     data[900:].save_to_disk(args.save_path+'test')


# def load_train_dataset(data_dir:str) -> Dataset:
#     return load_from_disk(data_dir)


class CustomImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(self.dataset[idx]['image'])
        image   = auxiliars.both_transforms(image=image)['image']
        low_resolution_image = auxiliars.lowres_transform(image=image)['image']
        high_resolution_image = auxiliars.highres_transform(image=image)['image']

        return {
            'ID': idx,
            'lr_image': low_resolution_image,
            'hr_image': high_resolution_image
        }


# if __name__=='__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--save_path", type=str)
#   download_dataset(parser.parse_args())

