import torch
import numpy as np
from skimage import transform
from torchvision.transforms import Normalize


class Normalizer(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, mask = sample
        image = image.sub_(self.mean).div_(self.std)
        mask = mask.sub_(self.mean).div_(self.std)
        return image, mask


class ToTensor(object):

    def __call__(self, sample):
        image, mask = sample
        return torch.from_numpy(image), torch.from_numpy(mask)


class FlipUD(object):

    def __call__(self, sample):
        image, mask = sample
        return (np.flipud(image), np.flipud(mask))


class FlipLR(object):

    def __call__(self, sample):
        image, mask = sample
        return (np.fliplr(image), np.fliplr(mask))


class Resize(object):


    def __init__(self, image_size, mask_size):
        self.image_size = image_size
        self.mask_size = mask_size

    def __call__(self, sample):
        image, mask = sample
        image = transform.resize(image, (self.image_size, self.image_size))
        mask = transform.resize(mask, (self.mask_size, self.mask_size))
        return image, mask


class RandomCrop(object):

    def __call__(self, sample):
        image, mask = sample
        c = np.random.randint(0, 20)
        return (self._crop(image,c), self._crop(mask,c))

    def _crop(self, image, c):
        image_width, _ = image.shape
        startx = image_width//2-(c//2)
        starty = image_width//2-(c//2)
        return image[starty:starty+image_width,startx:startx+image_width]