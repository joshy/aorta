import glob

import numpy as np
import torch
import torchvision
from skimage import io, transform
from torch.utils.data import Dataset


class AortaImageDataset(Dataset):
    """ Aorta Dataset helper class for pytorch. """

    def __init__(self, subdir='train', transform=None):
        aorta = '../data/' + subdir + '/**/**/mask*.png'
        cine = '../data/' + subdir + '/**/**/image*.png'
        self.pairs = list(zip(glob.glob(cine), glob.glob(aorta)))
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image, mask = self.pairs[idx]
        image = self._load_nifti(image)
        mask = self._load_nifti(mask)
        if self.transform:
            image, mask = self.transform((image, mask))
        return image, mask

    def _load_nifti(self, image_file):
        return io.imread(image_file, as_grey=True)
