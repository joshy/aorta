import glob

import nibabel as nib
import torch
import torchvision
from skimage import transform
from torch.utils.data import Dataset
import numpy as np

class AortaDataset(Dataset):
    """ Aorta Dataset helper class for pytorch. """

    def __init__(self, root_dir='./data'):
        aorta = './data/**/**/aorta*'
        cine = './data/**/**/4Ch*'
        pairs = list(zip(glob.glob(aorta), glob.glob(cine)))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mask, image = self.pairs[idx]
        return self.__get_zero_slice(image), self.__get_zero_slice(mask)

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def __get_zero_slice(self, image_file):
        """ Gets the first slice, transforms the array and returns it. """
        img = nib.load(image_file)
        img_data = img.get_data()
        slice = img_data[:, :, 0].T
        transformed = transform.resize(slice, (512, 512))
        return self._normalize(transformed)