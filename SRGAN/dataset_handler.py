from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import auxiliars
import numpy as np
import os
import argparse


class CustomImageDataset(Dataset):
    """Custom PyTorch dataset for handling low and high resolution image data.

    Args:
        dataset (Dataset): The input dataset containing image data.

    Methods:
        __getitem__: retrieve an item from the dataset, preprocess it and returns
        it as a dictionary
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves an item from the dataset, preprocess it and returns
        it as a dictionary.

        Args:
            idx (int): index of the dataset sample to process

        Returns:
            dict: {'ID':..,'lr_image':...,'hr_image':...}
        """
        image = np.array(self.dataset[idx]["image"])
        image = auxiliars.both_transforms(image=image)["image"]
        low_resolution_image = auxiliars.lowres_transform(image=image)["image"]
        high_resolution_image = auxiliars.highres_transform(image=image)["image"]

        return {
            "ID": idx,
            "lr_image": low_resolution_image,
            "hr_image": high_resolution_image,
        }
