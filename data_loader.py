import os
import glob
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import transform


def load():
    """ Gets the pairs of image file and mask file. """
    aorta = './data/**/**/aorta*'
    cine = './data/**/**/4Ch*'
    pairs = list(zip(glob.glob(aorta), glob.glob(cine)))
    print('Segmentations',len(pairs))
    return pairs


def get_zero_slice(image_file):
    """ Gets the first slice, transforms the array and returns it. """
    img = nib.load(image_file)
    img_data = img.get_data()
    return img_data[:,:,0].T


def show_image_and_mask(pair):
    """ Plots the image and mask together. """
    mask_file, image_file = pair
    first_slice = get_zero_slice(image_file)
    mask_slice = get_zero_slice(mask_file)
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,4))
    ax1.imshow(first_slice)
    ax1.set_title('MRI Image')
    ax2.imshow(mask_slice)
    ax2.set_title('Aorta Mask')


def rescale(image, size=128):
    """ Rescales a image to given size, default 128. """
    return transform.resize(image, (size, size))
