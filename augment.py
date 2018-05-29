import numpy as np


class FlipUD(object):

    def __call__(self, sample):
        image, mask = sample
        print(image.shape, mask.shape)
        return (np.flipud(image), np.flipud(mask))


class FlipLR(object):

    def __call__(self, sample):
        image, mask = sample
        return (np.fliplr(image), np.fliplr(mask))


class RandomCrop(object):

    def __call__(self, sample):
        image, mask = sample
        c = np.random.randint(0, 50)
        return (self._crop(image,c), self._crop(mask,c))

    def _crop(self, image, c):
        image_width, _ = image.shape
        startx = image_width//2-(c//2)
        starty = image_width//2-(c//2)
        return image[starty:starty+image_width,startx:startx+image_width]