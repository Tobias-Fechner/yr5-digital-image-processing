"""
Module to define filters and their computation algorithms. Used by handler.py.

Digital Image Processing 2020, Assignment 1/1, 20%
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import statistics

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
        descriptor = "Filter name: {},\nLinearity: {},\nMask size: {}\nKernel shown below where possible:".format(
            self.name,
            self.linearity,
            self.maskSize
        )

        plt.imshow(self.kernel, interpolation='none')

        return descriptor

    @staticmethod
    def assertTypes(maskSize, kernel):
        assert isinstance(maskSize, int)        # Mask size must be integer
        assert maskSize % 2 == 1                # Mask size must be odd
        assert isinstance(kernel, np.ndarray)   # kernel should be n-dimensional numpy array

    @abstractmethod
    def computePixel(self, sub):
        pass

    def convolve(self, img, padding=True):
        """
        This function which takes an image and a kernel and returns the convolution of them.
        :param padding: bool defines if padding is used
        :param img: numpy array of image to be filtered
        :return: numpy array of filtered image (image convoluted with kernel)
        """
        if padding:
            # Create padding for edges
            pad = int((self.maskSize - 1) / 2)
            print("Padding of {} pixels created.".format(pad))
        else:
            pad = 0
            print("No padding added.")

        # Flip the kernel up/down and left/right
        self.kernel = np.flipud(np.fliplr(self.kernel))

        # Create output array of zeros with same shape and type as img array
        output = np.zeros_like(img)

        # Add padding of zeros to the input image
        imgPadded = np.zeros((img.shape[0] + 2*pad, img.shape[1] + 2*pad))

        # Insert image pixel values into padded array
        imgPadded[pad:-pad, pad:-pad] = img

        # Loop over every pixel of padded image
        for col in range(img.shape[1]):
            for row in range(img.shape[0]):
                # Create sub matrix of mask size surrounding pixel under consideration
                sub = imgPadded[row: row+self.maskSize, col: col+self.maskSize]
                output[row, col] = self.computePixel(sub)

        return output

class Median(Filter):
    def __init__(self, maskSize):

        kernel = np.zeros((maskSize,maskSize))
        middle = int((maskSize-1)/2)
        kernel[middle, middle] = 1

        super().__init__(maskSize, kernel, name='median', linearity='non-linear')

    def computePixel(self, sub):
        return statistics.median(sub.flatten())

class Mean(Filter):
    """
    Effectively a low pass filter. Alternative kernel implemented in class LowPass(Filter).
    """
    def __init__(self, maskSize):
        kernel = np.ones((maskSize,maskSize))/(maskSize**2)
        super().__init__(maskSize, kernel, name='mean', linearity='linear')

    def computePixel(self, sub):
        try:
            assert self.kernel.sum() == 1
        except AssertionError:
            raise Exception("Sum of kernel weights for mean filter should equal 1. They equal {}!".format(self.kernel.sum()))
        # element-wise multiplication of the kernel and image pixel under consideration
        return (self.kernel * sub).sum()

class Gaussian(Filter):
    def __init__(self, sig):
        # Calculate mask size from sigma value. Ensures filter approaches zero at edges (always round up)
        maskSize = ceil((6 * sig) + 1)

        # TODO: implement mask size override? or scaling down of kernel values
        if maskSize % 2 == 0:
            maskSize += 1
        else:
            pass

        ax = np.linspace(-(maskSize - 1) / 2., (maskSize - 1) / 2., maskSize)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

        super().__init__(maskSize, kernel, name='gaussian', linearity='non-linear')

    def computePixel(self, sub):
        """
        Element-wise multiplication of the kernel and image pixel under consideration,
        accounting for normalisation to mitigate DC distortion effects.
        :param sub: sub matrix of image pixel under consideration and surrounding pixels within mask size.
        :return: product of sub matrix with kernel normalised by sum of kernel weights
        """
        return (self.kernel * sub).sum()/ self.kernel.sum()

class HighPass(Filter):
    def __init__(self, maskSize):

        # TODO: Make ratio of intensity reduction vs. increase configurable for both high and low pass
        kernel = np.full((maskSize, maskSize), -1/(maskSize**2))
        middle = int((maskSize-1)/2)
        kernel[middle, middle] = 1 - 1/(maskSize**2)
        #TODO: Check for high and low pass filter if they are non-linear or linear
        super().__init__(maskSize, kernel, name='high-pass', linearity='non-linear')

    def computePixel(self, sub):
        try:
            assert -0.01 < self.kernel.sum() < 0.01
        except AssertionError:
            raise Exception("Sum of high pass filter weights should be effectively zero.")

        return (self.kernel * sub).sum()

class LowPass(Filter):
    def __init__(self, maskSize):

        kernel = np.zeros((maskSize, maskSize))
        middle = int((maskSize-1)/2)
        kernel[middle, :] = 1/8
        kernel[:, middle] = 1/8
        kernel[middle, middle] = 1/2

        super().__init__(maskSize, kernel, name='low-pass', linearity='non-linear')

    def computePixel(self, sub):
        return (self.kernel * sub).sum()/ self.kernel.sum()