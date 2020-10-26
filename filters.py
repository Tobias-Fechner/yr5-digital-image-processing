"""
Module to define filters and their computation algorithms. Used by handler.py.

Digital Image Processing 2020, Assignment 1/1, 20%
"""

from abc import ABC, abstractmethod
from scipy.signal import convolve2d

class Filter(ABC):
    def __init__(self, name, linearity, maskSize):
        self.name = name
        self.linearity = linearity
        self.maskSize = maskSize

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

    @abstractmethod
    def apply(self):
        pass

class Median(Filter):
    def __init__(self, maskSize):
        assert isinstance(maskSize, int)
        super().__init__(name='median', linearity='non-linear', maskSize=maskSize)

    def apply(self):
        raise NotImplementedError

class Mean(Filter):
    def __init__(self, maskSize):
        assert isinstance(maskSize, int)
        super().__init__(name='mean', linearity='linear', maskSize=maskSize)

    def apply(self):
        raise NotImplementedError