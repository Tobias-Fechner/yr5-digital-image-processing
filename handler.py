"""
Module used to load image, apply filter, and show images in figures. Use in conjunction with filters and
edge-detector modules to complete assignment.

Digital Image Processing 2020, Assignment 1/1, 20%
"""

# Import packages used in code
import numpy as np
from PIL import Image
import logging
import filters
import matplotlib.pyplot as plt
from pathlib import Path
import os.path

# Initialise logging used to track info and warning messages
logging.basicConfig()

# Global variables used for ease of access to test images
FOETUS_PATH_ORIGINAL = ".\\images\\foetus.png"
NZJERS_PATH_ORIGINAL = ".\\images\\NZjers1.png"

def getImageAsArray(path, convertToGrey=True):
    """
    Function loads image from local file system using Pillow image loader and returns as numpy array.
    :param convertToGrey: bool parameter set to convert img to grey scale on loading. Default == True.
    :param path: file path to look to for image
    :return: both numpy array of image pixel values and original Pillow image data for optional use (likely will discard)
    """
    try:
        # Use library Pillow to open image data
        imgData = Image.open(path)
    except FileNotFoundError:
        raise Exception("No file found at that file path. Check it's there and try again. If error persists, check for special characters in file path.")

    # Convert to grey scale by default
    if convertToGrey:
        # Convert image to grey scale
        imgData.convert("L")
    else:
        pass

    # Convert to 2D numpy array and return
    return np.asarray(imgData), imgData

def plotFigs(images, imageFilter=None, filteredTitle=None, edgeTitle=None):
    """
    Function provides uniform plotting of figures to show results of filtering alongisde original image. Later, image
    edges can be displayed in additional third column, automatically determined by the number of images passed in.
    :param images: list of images to display - determines number of columns of sub plot
    :param imageFilter: filter object used to get filter name used in most image titles
    :param filteredTitle: filtered image title override, used to display more than default information in title
    :param edgeTitle: edge title override
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

    # Set dimensions of side-by-side image display
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    # Create axis for first figure, set title and show grey-scale original image on plot
    ax1 = fig.add_subplot(1,len(images),1)
    ax1.title.set_text('original')
    plt.imshow(images[0], cmap='gray')

    # Create axis for second figure, set title and show grey-scale filtered image on plot
    ax2 = fig.add_subplot(1,len(images),2)
    if filteredTitle:
        title = filteredTitle
    elif imageFilter:
        title = imageFilter.name + "_maskSize" + str(imageFilter.maskSize)
    else:
        title = 'filtered'
    ax2.title.set_text(title)
    plt.imshow(images[1], cmap='gray')

    # Create axis for third figure if more than 2 images available, set title and show grey-scale edge detections on plot
    if len(images) > 2:
        # display the new image
        ax3 = fig.add_subplot(1, len(images), 3)
        if edgeTitle:
            title = edgeTitle
        else:
            title = 'edge image'
        ax3.title.set_text(title)
        plt.imshow(images[2], cmap='gray')

    # Show all plots in Jupyter display
    plt.show(block=True)

def saveAll(img, imageFilter, saveFilter=True):
    """
    Function used to save all filter results to disk, including kernel plot, filter object state and images.
    :param img: filtered image
    :param imageFilter: filter object
    :param saveFilter: boolean used to configure if filter object attributes saved to text file
    :return: None
    """

    # Locate current directory, used for joining with relative paths
    currentDir = Path().absolute()
    root = str(currentDir) + '\\..\outputs\{}\maskSize_{}\\'.format(imageFilter.name, imageFilter.maskSize)

    # Create root path if not present
    if not os.path.exists(root):
        os.makedirs(root)

    # Create pillow image object from filtered image array
    img_PIL = Image.fromarray(img, 'L')

    # Save filtered image from pillow image object
    img_PIL.save(root+'filtered_image.png', 'PNG')
    print("Saved filtered image to... \n{}\n\n".format(root+'filtered_image.png'))

    # Save figure of kernel plot to image
    plt.imsave(root +'kernel_plot.png', imageFilter.kernel)
    print("Saved filtered image to... \n{}\n\n".format(root+'kernel.png'))

    if saveFilter:
        # Save filter attributes (including kernel as array.tolist()) to text file for traceability
        # Open text file with write permissions
        with open(root+'filter.txt', 'w') as f:
            # Retrieve attributes from list of filter object dictionary items as key-value pairs
            for k, v in imageFilter.__dict__.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                else:
                    pass
                # Write attribute field and values to text file
                f.write(''.join("filter.{} = {}\n".format(k, v)))
        print("Saved filter object attributes to... \n{}\n\n".format(root + 'filter.txt'))
    else:
        pass

