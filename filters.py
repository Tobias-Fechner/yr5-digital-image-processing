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
from tqdm import tqdm
import logging

logging.basicConfig()

def padImage(img, maskSize):
    """
    Function pads image in two dimensions. Pad size is dependant on mask shape and therefore both pads are
    currently always equal since we only use square mask sizes. Added pixels have intensity zero, 0.
    :param maskSize: used to calculate number of pixels to be added on  image
    :param img: img to be padded
    :return:
    """
    # Create padding for edges
    pad = ceil((maskSize - 1) / 2)

    assert isinstance(pad, int)

    # Add padding of zeros to the input image
    imgPadded = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad)).astype('uint8')

    # Insert image pixel values into padded array
    imgPadded[pad:-pad, pad:-pad] = img

    print("Padding of {} pixels created.".format(pad))

    return imgPadded

def scale(x, ceiling=255):
    """
    Function scales array between 0 and maximo
    :param x: array of values to be scaled
    :param ceiling: max values/ top of scale
    :return: scaled array
    """
    assert isinstance(x, np.ndarray)
    try:
        # Check min, max to avoid div 0 error
        assert x.max() != x.min()
    except AssertionError:
        # If all values the same, return array as it is (un-scaled)
        if np.all(x == x.flatten()[0]):
            return x
        # Otherwise, raise exception as any range of values should be scaled
        else:
            raise Exception("Can't scale as min and max are the same and will cause div(0) error but not "
                            "all values are the same in array. Printing array... ", x)

    return ceiling * (x - x.min()) / (x.max() - x.min())

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
            imgPadded = padImage(img, self.maskSize)
        else:
            imgPadded = img
            print("No padding added.")

        # Flip the kernel up/down and left/right
        self.kernel = np.flipud(np.fliplr(self.kernel))

        # Create output array of zeros with same shape and type as img array
        output = np.zeros_like(img)

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

class HistogramFilter(ABC):
    def __init__(self, maskSize, name):
        assert isinstance(maskSize, int)        # Mask size must be integer
        try:
            assert maskSize % 2 == 1            # Mask size must be odd
        except AssertionError:
            maskSize += 1
            pass

        # Mask size will always be odd
        self.maskSize = maskSize
        self.name = name

    def getHistogramWithCS(self, img):
        """
        Function takes in image as array of pixel intensities and generates a histogram and scaled cumulative sum
        :param img: numpy array of pixel intensities
        :return: histogram array and scaled cumulative sum of histogram
        """
        try:
            assert img.dtype == 'uint8'
        except AssertionError:
            img = img.astype('uint8')
            pass
        finally:
            assert img.dtype == 'uint8'

        # Create zeros array with as many items as bins to collect for
        histogram = np.zeros(256)

        # Loop through image and add pixel intensities to histogram
        for pixel in img.flatten():
            histogram[pixel] += 1

        csScaled = self.getCSScaled(histogram)

        return histogram.astype('uint8'), csScaled

    def filter(self, img, plotHistograms=True):
        imgFiltered, histogram, cs = self.compute(img)

        if plotHistograms:
            histogramNew, csNew = self.getHistogramWithCS(imgFiltered)
            self.plotHistograms(histogram, histogramNew, cs, csNew)
        else:
            pass

        return imgFiltered

    @staticmethod
    def getCSScaled(histogram):
        """
        Function returns cumulative sum of histogram scaled to range 0-255
        :param histogram: histogram of an image as array of len 256
        :return: scaled cumulative sum of histogram, of type int, scaled to 0-255
        """
        # Generate cumulative sum of histogram
        cs = np.cumsum(histogram.flatten())

        # Scale cumulative sum to be within range 0-255
        csScaled = scale(cs)

        # Return as type integer
        return csScaled.astype('uint8')

    @staticmethod
    def plotHistograms(histogram, histogramNew, cs, csNew):
        """
        Function plots overlaying histograms with cumulative sum to show change between original and filtered histogram.
        If no filtered histogram present, second series will be skipped.
        :param csNew:
        :param cs: cumulative sum of histogram
        :param histogramNew: histogram after filtering technique
        :param histogram: histogram of original image
        :return: None
        """
        # Set up figure
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(15)

        # Scale histograms so they overlay with scaled cumulative sums nicely and plot as bars with cs as lines
        plt.fill_between(np.arange(np.size(histogram)), scale(histogram), label='original_hist', alpha=0.4)
        plt.plot(cs, label='original_cs')
        try:
            plt.fill_between(np.arange(np.size(histogramNew)), scale(histogramNew), label='filtered_hist', alpha=0.4)
            plt.plot(csNew, label='filtered_cs')
        except ValueError:
            print("Only one histogram to plot.")
            pass

        plt.legend()
        plt.show()

    @staticmethod
    def interpolate(subBin, LU, RU, LB, RB, subX, subY):
        """

        :param subBin:
        :param LU:
        :param RU:
        :param LB:
        :param RB:
        :param subX:
        :param subY:
        :return:
        """
        subImage = np.zeros(subBin.shape)
        num = subX * subY
        for i in range(subX):
            inverseI = subX - i
            for j in range(subY):
                inverseJ = subY - j
                val = subBin[i, j].astype(int)
                subImage[i, j] = np.floor(
                    (inverseI * (inverseJ * LU[val] + j * RU[val]) + i * (inverseJ * LB[val] + j * RB[val])) / float(
                        num))
        return subImage

    @abstractmethod
    def compute(self, img):
        pass

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

class TruncateCoefficients(FourierFilter):
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

class Equalise(HistogramFilter):
    """
    This filter normalises the brightness whilst increasing the contrast of the image at the same time.
    """
    def __init__(self):
        super().__init__(None, name='histogram-equalise')

    def compute(self, img):
        histogram, cs = self.getHistogramWithCS(img)

        imgNew = cs[img.flatten()]

        return np.reshape(imgNew, img.shape), histogram, cs

class AHE(HistogramFilter):
    def __init__(self, maskSize=32):
        super().__init__(maskSize, name='adaptive-histogram-equalise')

    def compute(self, img, padding=True):
        """
        Function adds padding to image, then scans window over each pixel, calculating the histogram, finding the
        cumulative sum and using that to equalise the window around the pixel under consideration. The equalised pixel
        intensity is used in the filtered image output, for each pixel.
        :param img: numpy array of pixel intensities for original image
        :param padding: boolean specifying if padding is added to image (default:True)
        :return: returns filtered image with adaptive histogram equalised pixel intensities
        """
        # Get histogram and cumulative sum of filtered image
        histogramOriginal, csOriginal = self.getHistogramWithCS(img)

        if padding:
            imgPadded = padImage(img, self.maskSize)
        else:
            imgPadded = img
            print("No padding added.")

        # Create output array of zeros with same shape and type as img array
        imgFiltered = np.zeros_like(img)

        # Loop over every pixel of padded image
        for col in tqdm(range(img.shape[1])):
            for row in range(img.shape[0]):
                # Create sub matrix of mask size surrounding pixel under consideration
                sub = imgPadded[row: row+self.maskSize, col: col+self.maskSize]

                # Generate histogram and cumulative sum of sub matrix
                _, cs = self.getHistogramWithCS(sub)

                # Use histogram and cumulative sum to equalise the pixel intensities
                subNew = np.reshape(cs[sub.flatten()], sub.shape)

                # Update pixel under consideration with equalised pixel intensity
                middle = int((self.maskSize - 1) / 2)
                imgFiltered[row, col] = subNew[middle, middle]

        # Returns histogram and cumulative sum of original image for debugging purposes and to comply with
        # return pattern of filter function in parent class
        return imgFiltered, histogramOriginal, csOriginal

class SWAHE(HistogramFilter):
    def __init__(self, maskSize=32):
        super().__init__(maskSize, name='sliding-window-adaptive-histogram-equalise')

    def updateHistogramAndSub(self, histogram, sub, nextCol):
        for pixelSub, pixelAdd in zip(sub[:, 0], nextCol):
            histogram[pixelSub] -= 1
            histogram[pixelAdd] += 1

        sub = np.delete(sub, 0, axis=1)

        return histogram, np.append(sub, nextCol.reshape((self.maskSize, 1)), axis=1)

    def compute(self, img, padding=True):
        """
        Function implements same filter as AdaptiveEqualise but with sliding window computation method for faster
        computation.
        :param img: numpy array of pixel intensities for original image
        :param padding: boolean specifying if padding is added to image (default:True)
        :return: returns filtered image with adaptive histogram equalised pixel intensities
        """
        # Get histogram and cumulative sum of filtered image
        histogramOriginal, csOriginal = self.getHistogramWithCS(img)

        if padding:
            imgPadded = padImage(img, self.maskSize)
        else:
            imgPadded = img
            print("No padding added.")

        # Create output array of zeros with same shape and type as img array
        imgFiltered = np.zeros_like(img)

        # Loop over every pixel of *original* image
        for row in tqdm(range(img.shape[0])):
            # Create sub matrix of mask size surrounding pixel under consideration
            sub = np.array(imgPadded[row: row+self.maskSize, 0: 0+self.maskSize])

            # Generate histogram and cumulative sum of sub matrix
            histogram, cs = self.getHistogramWithCS(sub)

            # Use cumulative sum to equalise the pixel intensities
            subEqualised = np.reshape(cs[sub.flatten()], sub.shape)

            # Update pixel under consideration with equalised pixel intensity
            middle = int((self.maskSize - 1) / 2)
            imgFiltered[row, 0] = subEqualised[middle, middle]

            for col in range(img.shape[1]):
                try:
                    # Get next column of sub array in image
                    nextCol = imgPadded[row: row+self.maskSize, col+self.maskSize]
                except IndexError:
                    if col + self.maskSize == imgPadded.shape[1] + 1:
                        continue
                    else:
                        raise IndexError("Index error triggered unexpectedly when at column {}, row {}.".format(col, row))

                # Create sub matrix of mask size surrounding pixel under consideration
                histogram, sub = self.updateHistogramAndSub(histogram, sub, nextCol)

                # Get cumulative sum for updated histogram
                cs = self.getCSScaled(histogram)

                # Use histogram and cumulative sum to equalise the pixel intensities
                subEqualised = np.reshape(cs[sub.flatten()], sub.shape)

                # Update pixel under consideration with equalised pixel intensity
                middle = int((self.maskSize - 1) / 2)
                imgFiltered[row, col] = subEqualised[middle, middle]

        # Returns histogram and cumulative sum of original image for debugging purposes and to comply with
        # return pattern of filter function in parent class
        return imgFiltered, histogramOriginal, csOriginal

class CLAHE(HistogramFilter):
    def __init__(self, maskSize):
        super().__init__(maskSize, name='contrast-limited-adaptive-histogram-equalise')

    def compute(self, img):
        raise NotImplementedError

    @staticmethod
    def clahe(img, clipLimit, bins=128, maskSize=32):
        """
        Function performs clipped adaptive histogram equalisation on input image
        :param maskSize: size of kernel to scan over image
        :param img: input image as array
        :param clipLimit: normalised clip limit
        :param bins: number of gray level bins for histogram
        :return: return calhe image
        """
        if clipLimit == 1: return

        # Get number of rows and columns of img array
        row, col = img.shape
        # Allow min 128 bins
        bins = max(bins, 128)

        # Pad image to allow for integer number of kernels to fit in rows and columns
        subRows = ceil(row / maskSize)
        subCols = ceil(col / maskSize)

        # Get size of padding
        padX = int(maskSize * (subRows - row / maskSize))
        padY = int(maskSize * (subCols - col / maskSize))

        if padX != 0 or padY != 0:
            imgPadded = padImage(img, padX, padY)
        else:
            imgPadded = img
            print("No padding needed as the mask size of {} creates {} mini rows from the original image with {} rows."
                  "Likewise, {} mini columns from the original image with {} columns.".format(maskSize,subRows,row,subCols,col))

        noPixels = maskSize**2
        # xsz2 = round(kernelX / 2)
        # ysz2 = round(kernelY / 2)
        claheImage = np.zeros(imgPadded.shape)

        if clipLimit > 0:
            # Allow minimum clip limit of 1
            clipLimit = max(1, clipLimit * maskSize**2 / bins)
        else:
            # Convert any negative clip limit to 50
            clipLimit = 50

        # makeLUT
        print("...Make the LUT...")
        minVal = 0  # np.min(img)
        maxVal = 255  # np.max(img)

        # maxVal1 = maxVal + np.maximum(np.array([0]),minVal) - minVal
        # minVal1 = np.maximum(np.array([0]),minVal)

        binSz = np.floor(1 + (maxVal - minVal) / float(bins))
        LUT = np.floor((np.arange(minVal, maxVal + 1) - minVal) / float(binSz))

        # BACK TO CLAHE
        bins = LUT[img]
        print(bins.shape)
        # makeHistogram
        print("...Making the Histogram...")
        hist = np.zeros((subRows, subCols, bins))
        print(subRows, subCols, hist.shape)
        for i in range(subRows):
            for j in range(subCols):
                bin_ = bins[i * maskSize:(i + 1) * maskSize, j * maskSize:(j + 1) * maskSize].astype(int)
                for i1 in range(maskSize):
                    for j1 in range(maskSize):
                        hist[i, j, bin_[i1, j1]] += 1

        # clipHistogram
        print("...Clipping the Histogram...")
        if clipLimit > 0:
            for i in range(subRows):
                for j in range(subCols):
                    nrExcess = 0
                    for nr in range(bins):
                        excess = hist[i, j, nr] - clipLimit
                        if excess > 0:
                            nrExcess += excess

                    binIncr = nrExcess / bins
                    upper = clipLimit - binIncr
                    for nr in range(bins):
                        if hist[i, j, nr] > clipLimit:
                            hist[i, j, nr] = clipLimit
                        else:
                            if hist[i, j, nr] > upper:
                                nrExcess += upper - hist[i, j, nr]
                                hist[i, j, nr] = clipLimit
                            else:
                                nrExcess -= binIncr
                                hist[i, j, nr] += binIncr

                    if nrExcess > 0:
                        stepSz = max(1, np.floor(1 + nrExcess / bins))
                        for nr in range(bins):
                            nrExcess -= stepSz
                            hist[i, j, nr] += stepSz
                            if nrExcess < 1:
                                break

        # mapHistogram
        print("...Mapping the Histogram...")
        map_ = np.zeros((subRows, subCols, bins))
        # print(map_.shape)
        scale = (maxVal - minVal) / float(noPixels)
        for i in range(subRows):
            for j in range(subCols):
                sum_ = 0
                for nr in range(bins):
                    sum_ += hist[i, j, nr]
                    map_[i, j, nr] = np.floor(min(minVal + sum_ * scale, maxVal))

        # BACK TO CLAHE
        # INTERPOLATION
        print("...interpolation...")
        xI = 0
        for i in range(subRows + 1):
            if i == 0:
                subX = int(maskSize / 2)
                xU = 0
                xB = 0
            elif i == subRows:
                subX = int(maskSize / 2)
                xU = subRows - 1
                xB = subRows - 1
            else:
                subX = maskSize
                xU = i - 1
                xB = i

            yI = 0
            for j in range(subCols + 1):
                if j == 0:
                    subY = int(maskSize / 2)
                    yL = 0
                    yR = 0
                elif j == subCols:
                    subY = int(maskSize / 2)
                    yL = subCols - 1
                    yR = subCols - 1
                else:
                    subY = maskSize
                    yL = j - 1
                    yR = j
                UL = map_[xU, yL, :]
                UR = map_[xU, yR, :]
                BL = map_[xB, yL, :]
                BR = map_[xB, yR, :]
                # print("CLAHE vals...")
                subBin = bins[xI:xI + subX, yI:yI + subY]
                # print("clahe subBin shape: ",subBin.shape)
                subImage = HistogramFilter.interpolate(subBin, UL, UR, BL, BR, subX, subY)
                claheImage[xI:xI + subX, yI:yI + subY] = subImage
                yI += subY
            xI += subX

        if padX == 0 and padY != 0:
            return claheImage[:, :-padY]
        elif padX != 0 and padY == 0:
            return claheImage[:-padX, :]
        elif padX != 0 and padY != 0:
            return claheImage[:-padX, :-padY]
        else:
            return claheImage