import glob

import nibabel as nib
import torch
import torchvision
from skimage import transform
from torch.utils.data import Dataset
import numpy as np

class AortaDataset(Dataset):
    """ Aorta Dataset helper class for pytorch. """

    def __init__(self, subdir='train', transform=None):
        aorta = '../data/' + subdir + '/**/**/aorta*'
        cine = '../data/' + subdir + '/**/**/4Ch*'
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


    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


    def _groundtruth(self, mask, size):
        mask = self._get_zero_slice(mask, 388)
        foreground = np.where(mask > 0, 1, 0)
        background = np.where(foreground == 0, 1, 0)
        combined = np.stack((foreground, background))
        return combined


    def _load_nifti(self, image_file):
        nifti = nib.load(image_file)
        img_data = nifti.get_data()
        slice = img_data[:, :, 0].T
        return slice


    def _get_zero_slice(self, image_file, size):
        """ Gets the first slice, transforms the array and returns it. """
        nifti = self._load_nifti(image_file)
        img_data = nifti.get_data()

        slice = img_data[:, :, 0].T
        transformed = transform.resize(slice, (size, size))
        #return self._normalize(transformed)
        return transformed