"""
Module to define filters and their computation algorithms. Used by handler.py.

Digital Image Processing 2020, Assignment 1/1, 20%
"""

from abc import ABC
import numpy as np

class Filter(ABC):
    def __init__(self, maskSize, kernel, name, linearity):
        self.assertTypes(maskSize, kernel)
        self.name = name
        self.linearity = linearity
        self.maskSize = maskSize
        self.kernel = kernel

    def __str__(self):
        """
        Override built-in str function so a description of the filter is shown when you run print(filter), where
        filter is an instance of class Filter.
        :return: string describing filter instance.
        """
        descriptor = "Filter name: {},\nLinearity: {},\nMask size: {}\n".format(
            self.name,
            self.linearity,
            self.maskSize
        )
        return descriptor

    @staticmethod
    def assertTypes(maskSize, kernel):
        assert isinstance(maskSize, int)
        assert isinstance(kernel, np.ndarray)

    def convolve2D(self, img):
        """
        This function which takes an image and a kernel and returns the convolution of them.
        :param img: a numpy array of size [image_height, image_width].
        :return: a numpy array of size [image_height, image_width] (convolution output).
        """
        # Flip the kernel
        self.kernel = np.flipud(np.fliplr(self.kernel))
        # Create output array of zeros with same shape and type as img array
        output = np.zeros_like(img)

        # Add zero padding to the input image
        imgPadded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
        # TODO: check if this should be [1:-2, 1:-2]
        imgPadded[1:-1, 1:-1] = img

        # Loop over every pixel of the image
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                # element-wise multiplication of the kernel and the image
                # TODO: check why y+3, x+3
                output[y, x] = (self.kernel * imgPadded[y: y+3, x: x+3]).sum()

        return output

class Median(Filter):
    def __init__(self, maskSize):
        kernel = None
        super().__init__(maskSize, kernel, name='median', linearity='non-linear')

class Mean(Filter):
    def __init__(self, maskSize):
        kernel = np.ones((3,3))/9.0
        # TODO: Pass this into convolve function.
        super().__init__(maskSize, kernel, name='mean', linearity='linear')