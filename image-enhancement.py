import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import tkinter as tk

FOETUS_PATH_ORIGINAL = ".\\images\\foetus.png"
NZJERS_PATH_ORIGINAL = ".\\images\\NZjers1.png"

def updateImages():
    #x = slider1.get()
    #y = slider2.get()

    # # apply filter to images with new parameters - with applyFilter()
    # # save new images to disk
    # # load new images into GUI
    label1.configure(image=window.nzjers)
    label1.photo = window.nzjers
    print("image updated")

def applyFilter():
    raise NotImplementedError

# Instantiate new TKinter instance - Python's GUI builder
window = tk.Tk()

# Add two sliders to the GUI to interactively experiment with filter parameters
slider1 = tk.Scale(window, from_=0, to=1)
slider1.pack()
slider2 = tk.Scale(window, from_=0, to=1)
slider2.pack()

# Refresh images with new filters
button = tk.Button(window, text="Update", command=updateImages)
button.pack()

window.foetus = tk.PhotoImage(file=FOETUS_PATH_ORIGINAL)
window.nzjers = tk.PhotoImage(file=NZJERS_PATH_ORIGINAL)

# Create a canvas to load the images into
label1 = tk.Label(window, image=window.foetus)
label1.pack()

# Begin GUI loop to enable interaction with GUI window
window.mainloop()

# Think this is broken now
figures = {'foetus': FOETUS_PATH_ORIGINAL,
           'NZjers1': NZJERS_PATH_ORIGINAL}


# def plotFigs(figures, nrows=2, ncols=1):
#     """
#     Function to
#
#     :param figures: dictionary of {title, figure} pairs
#     :param nrows: number of columns wanted in the display
#     :param ncols: number of rows wanted in the display
#     :return:
#     """
#
#     fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
#
#     for i, title in enumerate(figures):
#         axes.ravel()[i].imshow(figures[title], cmap=plt.gray())
#         axes.ravel()[i].set_title(title)
#         axes.ravel()[i].set_axis_off()
#
#     plt.tight_layout()
#     #plt.colorbar()
#     #plt.hist()
#     plt.show()

# plotFigs(figures, 2, 1)