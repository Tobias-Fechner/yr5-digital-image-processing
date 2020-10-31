import cv2.cv2 as cv
import handler

def equalise(path):
    """
    CV2 histogram equalise function.
    :param path:
    :return:
    """
    img = cv.imread(path, 0)
    equ = cv.equalizeHist(img)
    handler.plotFigs([img, equ])

def CLAHE(path):
    """
    CV2 CLAHE function.
    :param path:
    :return:
    """
    img = cv.imread(path,0)
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    handler.plotFigs([img, cl1])