from datasets import load_dataset, load_from_disk, Dataset
import config


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
    



