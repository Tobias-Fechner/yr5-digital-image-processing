"""
Module used to load image, apply filter, and show images in figures. Use in conjunction with filters and
edge-detector modules to complete assignment.

Digital Image Processing 2020, Assignment 1/1, 20%
"""

import numpy as np
from PIL import Image
import logging
from IPython.display import display

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
        raise Exception("No file found at that file path. Check it's there and try again. "
                        "If error persists, check for special characters in file path.")

    if convertToGrey:
        # TODO: Convert to grey scale image
        raise NotImplementedError
    else:
        pass

    # Convert to 2D numpy array and return
    return np.asarray(imgData), imgData

def applyFilter(image, filtr):
    # TODO: Check filter present in filters.py module
    # TODO: Check parameters are correct
    # TODO: Convolve filter with image
    # TODO: Return filtered image.
    raise NotImplementedError

def plotFigs(images):
    """
    Simple function to display image(s) in notebook. Intended for use to see original vs filtered images.

    :return: None
    """
    try:
        assert isinstance(images, list)
    except AssertionError:
        if isinstance(images, np.ndarray):
            images = [images]
            pass
        else:
            raise Exception("Make sure you pass in either a single image as np ndarray or list of images as np ndarray.")

    for img in images:
        img_PIL = Image.fromarray(img, 'L')
        display(img_PIL)