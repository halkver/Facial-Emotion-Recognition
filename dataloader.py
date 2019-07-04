from torch.utils.data.dataset import Dataset
import pandas as pd
from preprocess import preprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Dataset(Dataset):
    """
    FER2013 Dataset
    Args:
        set_type: string, optional - Training / PublicTest / PrivateTest
        transform: callable, optional - Transform that uses an PIL image
        path: string, optional - Path to dataset
    """

    def __init__(self, set_type='Training', transform=None, path='data/fer2013.csv'):
        self.transform = transform
        self.all_data = preprocess(path, set_type)
        self.data = np.stack(self.all_data.pixels.values).reshape((len(self.all_data), 48, 48))
        self.labels = self.all_data.emotion.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns item at index
        Args:
            index: int - Index
        Returns:
            (image, target): tuple - Image with its target
        """
        image = Image.fromarray(np.uint8(self.data[index]), 'L')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]
