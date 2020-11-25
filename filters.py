"""
Main module to define filters and their computation algorithms.

These design requirements have been achieved by implementing abstract base classes (ABC) for each ‘family’ of filter
(spatial, Fourier, histogram), and several associated child classes for the filters themselves.
The functionality common to each filter within a given family has been defined in the base class and then
inherited by each child class. Any method implemented as an abstract method, in which it is not defined within
the base class but only declared, must be defined in the child class as a requirement, as is the case
with the method ‘compute’. This framework-like approach both enforces and supports the level of granularity
required to uniquely define the computation method for a given filtering technique.

Digital Image Processing 2020, Assignment 1/1, 20%
"""

# Import relevant packages
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import statistics
from tqdm import tqdm
import logging

# Initialise logging used to track info and warning messages
logging.basicConfig()

def padImage(img, maskSize):
    """
    Function pads image in two dimensions. Pad size is dependant on mask shape and therefore both pads are
    currently always equal since we only use square mask sizes. Added pixels have intensity zero, 0.
    :param maskSize: used to calculate number of pixels to be added on  image
    :param img: image to be padded
    :return: padded array of pixel intensities
    """
    # Calculate number of pixels required for padding
    pad = ceil((maskSize - 1) / 2)

    assert isinstance(pad, int)

    # Add pad number of rows and columns of zeros around the sides of the input image array
    imgPadded = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad)).astype('uint8')

    # Insert image pixel values into padded array
    imgPadded[pad:-pad, pad:-pad] = img

    # Log success result to console
    logging.info("Padding of {} pixels created.".format(pad))

    return imgPadded

def scale(x, ceiling=255):
    """
    Function scales n-dimensional array of values between zero to max value, cieling
    :param x: n-dimensional array of values to be scaled
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

    # Return array with values scaled between zero and max
    return ceiling * (x - x.min()) / (x.max() - x.min())

class SpatialFilter(ABC):
    """
    Base class for all spatial filters, inherited by any spatial filter and containing all methods and attributes
    common to operation of all spatial filters.
    """
    def __init__(self, maskSize, kernel, name, linearity):
        """
        Object initialisation override used to assign parameters passed on creation of new class instance to object attributes.
        :param maskSize: mask size used to scan over pixels during convolution to detect surrounding pixel intensities (aka window size)
        :param kernel: kernel of weights used to multiply with pixel intensities to calculate pixel update value
        :param name: meta information - name of filter
        :param linearity: meta information - linearity of filter
        """
        self.assertTypes(maskSize, kernel)
        self.name = name
        self.linearity = linearity
        self.maskSize = maskSize
        self.kernel = kernel

    def __str__(self):
        """
        Override built-in str function so a description of the filter is shown when you run print(filter), where
        filter is an instance of class Filter.
        :return: generated string describing filter instance/ object state
        """
        # Combine various object attributes into descriptive string to be displayed
        descriptor = "Filter name: {},\nLinearity: {},\nMask size: {}\nKernel shown below where possible:".format(
            self.name,
            self.linearity,
            self.maskSize
        )

        # Generate plot of kernel weights, used to visualise kernel weight distribution
        plt.imshow(self.kernel, interpolation='none')

        return descriptor

    @staticmethod
    def assertTypes(maskSize, kernel):
        """
        Static method used for basic type checking during filtering computation.
        :param maskSize: filter window/ mask size
        :param kernel: filter kernel of weights
        :return: None
        """
        assert isinstance(maskSize, int)        # Mask size must be integer
        assert maskSize % 2 == 1                # Mask size must be odd
        assert isinstance(kernel, np.ndarray)   # kernel should be n-dimensional numpy array

    @abstractmethod
    def compute(self, sub):
        """
        Abstract method declared here in base class and later defined in child classes that must as a rule inherit this method.
        This is the krux of the ABC design approach - each filter will and must uniquely implement its own computation method to
        calculate the pixel update value based on its intended filtering function.
        :param sub: the sub matrix/ window of pixel values generated from convolution of the window with image
        :return: pixel update value
        """
        pass

    def convolve(self, img, padding=True):
        """
        Convolution of filter object's kernel over the image recieved as a parameter to this function.
        :param padding: boolean used to configure the addition of zero-padding to image.
        :param img: n-dimensional numpy array of original image pixel values that will each be updates during filtering i.e the original image data
        :return: numpy array of dimension equal to original image array with updated pixel values i.e. the filtered image data
        """
        # If padding required, create padding, else original image stored as padded image
        if padding:
            imgPadded = padImage(img, self.maskSize)
        else:
            imgPadded = img
            logging.warning("No padding added. This may mean the first/ last pixels of each row may not be filtered.")

        # Flip the kernel up/down and left/right
        self.kernel = np.flipud(np.fliplr(self.kernel))

        # Create output array of zeros with same shape and type as original image data
        output = np.zeros_like(img)

        # Iterate over every column in that row
        for col in tqdm(range(img.shape[1])):

            # Iterate over every row in the image
            for row in range(img.shape[0]):

                # Create sub matrix of mask size surrounding pixel under consideration
                sub = imgPadded[row: row+self.maskSize, col: col+self.maskSize]

                # Store the updated pixel intensity (returned from the filter's own computation method) in the filtered image array
                output[row, col] = self.compute(sub)

        return output

class HistogramFilter(ABC):
    """
    Base class for all histogram filters, inherited by any histogram filter and containing all methods and attributes
    common to operation of all histogram filters.
    """
    def __init__(self, maskSize, name):
        """
        Object initialisation override used to assign parameters passed on creation of new class instance to object attributes.
        :param maskSize: mask size used to scan over pixels during convolution to detect surrounding pixel intensities (aka window size)
        :param name: meta information - name of filter
        """
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
        Function takes in image as an n-dimensional array of pixel intensities and generates a histogram and scaled cumulative sum
        :param img: numpy array of pixel intensities
        :return: histogram array and scaled cumulative sum of histogram
        """
        # Catch errors for wrong data type, allowing for one exception by casting to integer on first exception
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
        """
        Primary access point from external code for any histogram filter. Equivalent to convolve for Spatial filters.
        Function computes and returns filtered image.
        :param img: original image data
        :param plotHistograms: boolean used to configure if a plot of original and updated histograms should be displayed to
        Jupyter notebook or not.
        :return: filtered image
        """
        # Call computation method unique to each filter implementation.
        imgFiltered, histogram, cs = self.compute(img)

        # Plot histograms if required
        if plotHistograms:
            # Generate histogram and cumulative sum for filtered image
            histogramNew, csNew = self.getHistogramWithCS(imgFiltered)
            # Plot histograms for display in notebook
            self.plotHistograms(histogram, histogramNew, cs, csNew)
        else:
            pass

        # Return filtered image
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
        :param csNew: cumulative sum of filtered image histogram values
        :param cs: cumulative sum of original image histogram values
        :param histogramNew: histogram after filter has been applied
        :param histogram: histogram of original image
        :return: None
        """
        # Set dimensions of figure
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
            logging.info("Only one histogram to plot.")
            pass

        # Add legend and show plot of histograms
        plt.legend()
        plt.show()

    @abstractmethod
    def compute(self, img):
        """
        Abstract method declared here in base class and later defined in child classes that must as a rule inherit this method.
        This is the krux of the ABC design approach - each filter will and must uniquely implement its own computation method to
        calculate the pixel update value based on its intended filtering function.
        :param img: the n-dimensional array of pixel values that represent the original image data
        :return: pixel update value
        """
        pass

class Median(SpatialFilter):
    def __init__(self, maskSize):

        # Arbitrary kernel weights assigned since kernel is not used
        super().__init__(maskSize, np.zeros((maskSize,maskSize)), name='median', linearity='non-linear')

    def compute(self, sub):
        # Python's statistics library is used to compute the statistical median of
        # the flattened pixel array
        return statistics.median(sub.flatten())

class AdaptiveWeightedMedian(SpatialFilter):
    def __init__(self, maskSize, constant, centralWeight):

        # Create 1D array of linearly distributed values with given start/ stop values and a step size of maskSize
        ax = np.linspace(-(maskSize - 1) / 2., (maskSize - 1) / 2., maskSize)

        # Create coordinate grid using 1D linspace array
        xx, yy = np.meshgrid(ax, ax)

        # Finally, create kernel of weight corresponding to distance from centre using pythagoras theorem
        kernel = np.sqrt(np.square(xx) + np.square(yy))

        # set max weight, used for centre of kernel, and constant used in formula
        self.constant = constant
        self.centralWeight = centralWeight

        super().__init__(maskSize, kernel, name='adaptive-weighted-median', linearity='non-linear')

    def compute(self, sub):
        # Calculate the standard deviation and mean of sub matrix
        std = np.std(sub)
        mean = np.mean(sub)

        if mean == 0:
            mean = 1
        else:
            pass

        # Create matrix of weights based on sub matrix, using formula for adaptive weighted median filter
        weights = self.centralWeight - self.constant*std*np.divide(self.kernel, mean)

        # Identify any negative weights in boolean array
        mask = weights < 0
        # Use as inverse mask truncate negative weights to zero to ensure low pass characteristics
        weights = np.multiply(np.invert(mask), weights)

        # Use list comprehension to pair each element from sub matrix with respective weighting in tuple
        # and sort based on sub matrix values/ pixel intensities
        pairings = sorted((pixelIntensity, weight) for pixelIntensity, weight in zip(sub.flatten(), weights.flatten()))

        # Calculate where median position will be
        medIndex = ceil((np.sum(weights) + 1)/ 2)
        cs = np.cumsum([pair[1] for pair in pairings])
        medPairIndex = np.searchsorted(cs, medIndex)

        # Return median of list of weighted sub matrix values
        return pairings[medPairIndex][0]

class Mean(SpatialFilter):
    """
    Effectively a blurring filter. Alternative kernel implemented in class LowPass(Filter).
    """
    def __init__(self, maskSize):

        # Kernel weights defined as one over the number of weights, thus summing to one
        kernel = np.ones((maskSize,maskSize))/(maskSize**2)

        # Ensure sum of mean kernel weights is essentially 1
        try:
            assert kernel.sum() == 1
        except AssertionError:
            if abs(1 - kernel.sum()) < 0.01:
                pass
            else:
                raise Exception("Sum of kernel weights for mean filter should equal 1. They equal {}!".format(kernel.sum()))

        super().__init__(maskSize, kernel, name='mean', linearity='linear')

    def compute(self, sub):
        # element-wise multiplication of the kernel and image pixel under consideration
        return np.sum(np.multiply(self.kernel, sub))

class TrimmedMean(SpatialFilter):
    """
    Can be used to discard a number of outliers from the higher and lower ends of the retrieved sub matrix of pixel values.
    """
    def __init__(self, maskSize, trimStart=1, trimEnd=1):

        # Same as the mean filter, kernel weights defined as one over the number of weights, thus summing to one
        kernel = np.ones((maskSize,maskSize))/(maskSize**2)

        # Ensure sum of weights equals one
        try:
            assert kernel.sum() == 1
        except AssertionError:
            if abs(1 - kernel.sum()) < 0.01:
                pass
            else:
                raise Exception("Sum of kernel weights for mean filter should equal 1. They equal {}!".format(kernel.sum()))

        # Assign trim parameters as attributes specific to this class for use in computation
        self.trimStart = trimStart
        self.trimEnd = trimEnd
        super().__init__(maskSize, kernel, name='trimmed-mean', linearity='linear')

    def compute(self, sub):

        # Flatten sub matrix
        trimmedSub = list(sub.flatten())

        # Index a specified number of elements from either end of the flattened array
        # Return mean of this selection of elements
        return np.mean(trimmedSub[self.trimStart:-self.trimStart])

class Gaussian(SpatialFilter):
    def __init__(self, sig):
        # Calculate mask size from sigma value. Ensures filter approaches zero at edges (always round up)
        maskSize = ceil((6 * sig) + 1)

        # Ensure mask size is odd
        if maskSize % 2 == 0:
            maskSize += 1
        else:
            pass

        # Create kernel with weights representing gaussian distribution with input standard deviation
        # Create 1D array of linearly distributed values with given start/ stop values and a step size of maskSize
        ax = np.linspace(-(maskSize - 1) / 2., (maskSize - 1) / 2., maskSize)

        # Create coordinate grid using 1D linspace array
        xx, yy = np.meshgrid(ax, ax)

        # Finally, create kernel using gaussian distribution formula
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

        super().__init__(maskSize, kernel, name='gaussian', linearity='linear')

    def compute(self, sub):
        """
        Element-wise multiplication of the kernel and image pixel under consideration,
        accounting for normalisation to mitigate DC distortion effects.
        :param sub: sub matrix of image pixel under consideration and surrounding pixels within mask size.
        :return: product of sub matrix with kernel normalised by sum of kernel weights
        """
        return np.sum(np.multiply(self.kernel, sub))/ self.kernel.sum()

class Sharpening(SpatialFilter):
    """
    High pass filter to have sharpening effect on image.
    """
    def __init__(self, maskSize):

        # Create kernel of negative one over the square of mask size
        kernel = np.full((maskSize, maskSize), -1)

        # Set centre pixel to positive fraction such that kernel weights sum to zero
        middle = int((maskSize-1)/2)
        kernel[middle, middle] = maskSize**2 - 1

        # Divide all elements by the number of elements in the window
        kernel = np.divide(kernel, maskSize**2)

        super().__init__(maskSize, kernel, name='high-pass', linearity='linear')

    def compute(self, sub):

        # Ensure sum of kernel weights is effectively zero
        try:
            assert -0.01 < np.sum(self.kernel) < 0.01
        except AssertionError:
            raise Exception("Sum of high pass filter weights should be effectively 0.")

        # Perform element-wise multiplication of kernel and window contents, then sum
        return np.sum(np.multiply(self.kernel, sub))

class LowPass(SpatialFilter):
    def __init__(self, maskSize, middleWeight=1/2, otherWeights=1/8):

        kernel = np.zeros((maskSize, maskSize))
        middle = int((maskSize-1)/2)
        kernel[middle, :] = otherWeights
        kernel[:, middle] = otherWeights
        kernel[middle, middle] = middleWeight

        super().__init__(maskSize, kernel, name='low-pass', linearity='non-linear')

    def compute(self, sub):
        return (self.kernel * sub).sum()/ self.kernel.sum()

class Equalise(HistogramFilter):
    """
    This filter normalises the brightness whilst increasing the contrast of the image at the same time.
    """
    def __init__(self):
        super().__init__(3, name='histogram-equalise')

    def compute(self, img):
        # Generate histogram and cumulative sum of original image
        histogram, cs = self.getHistogramWithCS(img)

        # Index pixel values from flattened original image at each value of the cumulative sum
        imgNew = cs[img.flatten()]

        # Return the image with evenly distributed pixel intensities with the same dimensions as original image
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
            logging.info("No padding added.")

        # Create output array of zeros with same shape and type as img array
        imgFiltered = np.zeros_like(img)

        # Loop over every pixel of padded image
        for row in tqdm(range(img.shape[0])):
            for col in range(img.shape[1]):

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

        # Pair pixels in the corresponding rows of the trailing and next columns
        for pixelSub, pixelAdd in zip(sub[:, 0], nextCol):
            # Subtract 1 from the histogram at the occurrence of each pixel intensity in the trailing column
            histogram[pixelSub] -= 1

            # Add one for each pixel intensity occurrence in the next kernel window column
            histogram[pixelAdd] += 1

        # Drop the trailing column of the sub matrix
        sub = np.delete(sub, 0, axis=1)

        # Return the histogram and sub matrix with next column appended
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
            logging.info("No padding added.")

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
                    
                    # Allow index error due to it being the last row in the row.
                    # Favoured computationally over running an if statement during each iteration
                    if col + self.maskSize <= imgPadded.shape[1] + 1:
                        continue
                    else:
                        raise IndexError("Index error triggered unexpectedly when at column {}, row {}.\n"
                                         "mask size = {}\n"
                                         "col+self.maskSize = {}\n"
                                         "imgPadded.shape[1] = {}\n".format(col, row, self.maskSize, col+self.maskSize, imgPadded.shape[1]))

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

