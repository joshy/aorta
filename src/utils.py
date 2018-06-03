import nibabel as nib
import matplotlib.pyplot as plt



def show_image_and_mask(image, mask):
    """ Plots the image and mask together. """
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,4))
    ax1.imshow(image)
    ax1.set_title('MRI Image')
    ax2.imshow(mask)
    ax2.set_title('Aorta Mask')
