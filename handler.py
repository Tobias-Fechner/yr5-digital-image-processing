"""
Module used to load image, apply filter, and show images in figures. Use in conjunction with filters and
edge-detector modules to complete assignment.

Digital Image Processing 2020, Assignment 1/1, 20%
"""

import numpy as np
from PIL import Image
import logging
from IPython.display import display
import filters
import matplotlib.pyplot as plt
from pathlib import Path
import os.path

logging.basicConfig()

FOETUS_PATH_ORIGINAL = ".\\images\\foetus.png"
NZJERS_PATH_ORIGINAL = ".\\images\\NZjers1.png"
FOETUS_PATH_FILTERED = ".\\images\\foetus-filtered.png"
NZJERS_PATH_FILTERED = ".\\images\\NZjers1-filtered.png"

def getImageAsArray(path, convertToGrey=True):
    """
    Function loads image from local file system using Pillow image loader and returns as numpy array.
    :param convertToGrey: bool parameter set to convert img to grey scale on loading. Default == True.
    :param path: file path to look to for image
    :return: both numpy array of image pixel values and original Pillow image data for optional use (likely will discard)
    """
    try:
        # Use Pillow to load image data
        imgData = Image.open(path)
    except FileNotFoundError:
        raise Exception("No file found at that file path. Check it's there and try again. If error persists, check for special characters in file path.")

    if convertToGrey:
        imgData.convert("L")
    else:
        pass

    # Convert to 2D numpy array and return
    return np.asarray(imgData), imgData

def plotFigs(images):
    """
    Simple function to display image(s) in notebook. Intended for use to see original vs filtered images.

    :return: None
    """
    # Check types and correct common error if necessary
    try:
        assert isinstance(images, list)
    except AssertionError:
        if isinstance(images, np.ndarray):
            images = [images]
            pass
        else:
            raise Exception("Make sure you pass in either a single image as np ndarray or list of images as np ndarray.")

    for img in images:
        # Convert array to geyscale pillow image object
        img_PIL = Image.fromarray(img, 'L')
        display(img_PIL)

def saveAll(img, filtr):
    """
    Function to save all figures relevant to report. Currently filtered image and plot of kernel.
    :return:
    """
    assert isinstance(filtr, filters.Filter)

    currentDir = Path().absolute()
    root = str(currentDir) + '\\..\outputs\{}\maskSize_{}\\'.format(filtr.name, filtr.maskSize)

    if not os.path.exists(root):
        os.makedirs(root)

    # Create pillow image object from filtered image array
    img_PIL = Image.fromarray(img, 'L')

    # Save filtered image from pillow image object
    img_PIL.save(root+'filtered_image.png', 'PNG')
    print("Saved filtered image to {}".format(root+'filtered_image.png'))

    # TODO: Make saved image of plot larger. Currently will be tiny if mask size is eg 9x9.
    # Save figure of kernel plot to image
    plt.imsave(root+'kernel_plot.png', filtr.kernel)
    print("Saved filtered image to {}".format(root+'kernel.png'))

    # Save filter attributes (including kernel as array.tolist()) to text file for traceability
    with open(root+'filter.txt', 'w') as f:
        for k, v in filtr.__dict__.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            else:
                pass
            f.write(''.join("filter.{} = {}\n".format(k, v)))
    print("Saved filter object attributes to {}".format(root + 'filter.txt'))