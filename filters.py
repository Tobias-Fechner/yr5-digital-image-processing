"""
Module to define filters and their computation algorithms. Used by handler.py.

Digital Image Processing 2020, Assignment 1/1, 20%
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
import statistics
from scipy import fftpack
from matplotlib.colors import LogNorm

class SpatialFilter(ABC):
    # TODO: implement standard mask shapes of square or cross and implement kernel creation based on this for each filter
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

class FourierFilter:
    def fft2D_scipy(self, img, plot=False):
        """
        Function transforms image into Fourier domain
        :param plot: bool to configure plotting of fourier spectum. default=False
        :param img: image to be transformed
        :return: image in fourier domain/ fourier spectrum of image
        """
        imgFFT = fftpack.fft2(img)
        if plot: self.plotFourierSpectrum(imgFFT)
        return imgFFT

    @staticmethod
    def dft(x): # not in use
        """
        Function computes the discrete fourier transform of a 1D array
        :param x: input array, 1 dimensional
        :return: np.array of fourier transformed input array
        """
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp((-2j * np.pi * k * n) / N)
        return np.dot(M, x)

    def fft(self, x):
        """
        Function recursively implements 1D Cooley-Turkey fast fourier transform
        :param x: input array, 1 dimensional
        :return: np.array of fourier transformed input array
        """
        x = np.array(x, dtype=float)
        N = x.shape[0]

        if N % 2 > 0:
            raise ValueError("size of x must be a power of 2")
        elif N <= 32:
            return self.dft(x)
        else:
            X_even = self.fft(x[::2])
            X_odd = self.fft(x[1::2])
            factor = np.exp((-2j * np.pi * np.arange(N)) /N)
            return np.concatenate([X_even + factor[:N / 2] * X_odd,
                                   X_even + factor[N / 2:] * X_odd ])


    def fft2D(self, x):
        """
        Function recursively implements 1D Cooley-Turkey fast fourier transform
        :param x: input array, 1 dimensional
        :return: np.array of fourier transformed input array
        """
        x = np.array(x, dtype=float)
        xRot = x.T

        self.fft(x)

    @staticmethod
    def inverseFFT_scipy(img):
        return fftpack.ifft2(img).real

    @staticmethod
    def plotFourierSpectrum(imgFFT):
        """
        Function displays fourier spectrum of image that has been fourier transformed
        :param imgFFT: fourier spectrum of img
        :return: None
        """
        plt.figure()
        plt.imshow(np.abs(imgFFT), norm=LogNorm(vmin=5))
        plt.colorbar()
        plt.title('Fourier Spectrum')


class Median(SpatialFilter):
    def __init__(self, maskSize):

        kernel = np.zeros((maskSize,maskSize))
        middle = int((maskSize-1)/2)
        kernel[middle, middle] = 1

        super().__init__(maskSize, kernel, name='median', linearity='non-linear')

    def computePixel(self, sub):
        return statistics.median(sub.flatten())

class Mean(SpatialFilter):
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

class Gaussian(SpatialFilter):
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

        super().__init__(maskSize, kernel, name='gaussian', linearity='linear')

    def computePixel(self, sub):
        """
        Element-wise multiplication of the kernel and image pixel under consideration,
        accounting for normalisation to mitigate DC distortion effects.
        :param sub: sub matrix of image pixel under consideration and surrounding pixels within mask size.
        :return: product of sub matrix with kernel normalised by sum of kernel weights
        """
        return (self.kernel * sub).sum()/ self.kernel.sum()

class HighPass(SpatialFilter):
    """
    High pass filter to have sharpening effect on image.
    """
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

class LowPass(SpatialFilter):
    def __init__(self, maskSize):

        kernel = np.zeros((maskSize, maskSize))
        middle = int((maskSize-1)/2)
        kernel[middle, :] = 1/8
        kernel[:, middle] = 1/8
        kernel[middle, middle] = 1/2

        super().__init__(maskSize, kernel, name='low-pass', linearity='non-linear')

    def computePixel(self, sub):
        return (self.kernel * sub).sum()/ self.kernel.sum()

class FFT_TruncateCoefficients(FourierFilter):
    def __init__(self, keep=0.1):
        self.keep = keep

    def compute(self, img, plot=False):
        # Get fourier transform of image
        imgFFT = self.fft2D_scipy(img, plot=plot)

        # Call ff a copy of original transform
        imgFFT2 = imgFFT.copy()

        # Get shape of image: rows and columns
        row, col = imgFFT2.shape

        # Set all rows and cols to zero not within the keep fraction
        imgFFT2[ceil(row*self.keep):floor(row*(1-self.keep)), :] = 0
        imgFFT2[:, ceil(col*self.keep):floor(col*(1-self.keep))] = 0

        if plot: self.plotFourierSpectrum(imgFFT2)

        return self.inverseFFT_scipy(imgFFT2)

class HistogramFilter:
    @staticmethod
    def getHistogram(img):
        # Create zeros array with as many items as bins to collect for
        histogram = np.zeros(256)

        # Loop through image and add pixel intensities to histogram
        for pixel in img.flatten():
            histogram[pixel] += 1

        cumSum = HistogramFilter.cumSum(histogram)
        cumSumNormalised = HistogramFilter.cumSumNormalise(cumSum)

        return histogram, cumSumNormalised

    @staticmethod
    def cumSum(histogram):
        return np.cumsum(histogram.flatten())

    @staticmethod
    def cumSumNormalise(cumSum):
        # normalize the cumsum to range 0-255
        csNormalised = 255 * (cumSum - cumSum.min()) / (cumSum.max() - cumSum.min())

        # cast it to uint8
        return csNormalised.astype('uint8')

    @staticmethod
    def histEqualise(img, cs):
        imgNew = cs[img.flatten()]

        return np.reshape(imgNew, img.shape)