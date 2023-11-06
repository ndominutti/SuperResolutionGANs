from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from . import auxiliars
import numpy as np
import os


def download_dataset():
    """
    Load datasets from Hugging Face and save them to disk
    
    Args:
      data_path(str): HF path to the XNLI dataset
    """
    data = load_dataset("satellite-image-deep-learning/SODA-A", split='train')
    
    data.shuffle(seed=1)
    data[:2000].save_to_disk(f'data/train')
    val[2000:].save_to_disk(f'data/eval')


def load_train_dataset(data_dir:str) -> Dataset:
    return load_from_disk(data_dir)


def process_train_dataset(dataset:Dataset) -> Dataset:
    dataset = dataset.map(
        _preprocessing_job,
        batched=True,
    )
    return dataset

def _preprocessing_job(img_file):
    image = np.array(img_file['image'])
    image = auxiliars.both_transforms(image=image)["image"]
    high_res = auxiliars.highres_transform(image=image)["image"]
    low_res = auxiliars.lowres_transform(image=image)["image"]
    return low_res, high_res


if __name__=='__main__':
    d = load_train_dataset('data/train/')
    process_train_dataset(d)
    

# class DataDIR(Dataset):
#     def __init__(self, root_dir):
#         super(DataDIR, self).__init__()
#         self.data = []
#         self.root_dir = root_dir
#         self.class_names = os.listdir(root_dir)

#         for index, name in enumerate(self.class_names):
#             files = os.listdir(os.path.join(root_dir, name))
#             self.data += list(zip(files, [index] * len(files)))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         img_file, label = self.data[index]
#         root_and_dir = os.path.join(self.root_dir, self.class_names[label])

#         image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
#         image = config.
#         high_res = config.highres_transform(image=image)["image"]
#         low_res = config.lowres_transform(image=image)["image"]
#         return low_res, high_res
    



