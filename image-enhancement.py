import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import logging
import techniques
import inspect

logging.basicConfig()

FOETUS_PATH_ORIGINAL = ".\\images\\foetus.png"
NZJERS_PATH_ORIGINAL = ".\\images\\NZjers1.png"
FOETUS_PATH_FILTERED = ".\\images\\foetus-filtered.png"
NZJERS_PATH_FILTERED = ".\\images\\NZjers1-filtered.png"

def getImageAsArray(path):
    try:
        # Use Pillow to load image data
        imgData = Image.open(path)

        # Convert to 2D numpy array and return
        return np.asarray(imgData)
    except FileNotFoundError:
        logging.error("File not found, returning empty array.")
        return np.array([])

def updateImages():

    filterType, maskSize, weightings, target = getParamsFromGUI()
    #x = slider1.get()
    #y = slider2.get()
    #z = dropdown.get() # to get filter type to be applied
    # filterResult = techniques.meanFilter(image, mask)
    # apply filter to images with new parameters - with applyFilter()


    # save new images to disk
    # load new images into GUI
    labelFoetusFiltered.configure(image=master.foetusFiltered)
    labelFoetusFiltered.photo = master.foetusFiltered
    print("image updated")

def applyFilter():
    raise NotImplementedError

def getParamsFromGUI():
    raise NotImplementedError
    #filterType =


#### LOAD IMAGES TO ARRAYS
# foetusOriginal = getImageAsArray(FOETUS_PATH_ORIGINAL)
# nzjersOriginal = getImageAsArray(NZJERS_PATH_ORIGINAL)
# foetusFiltered = getImageAsArray(FOETUS_PATH_FILTERED)
# nzjersFiltered = getImageAsArray(NZJERS_PATH_FILTERED)



#### CREATE GUI
# Instantiate new TKinter instance - Python's GUI builder
master = tk.Tk()

# Create frames to pack elements into
frame0 = tk.Frame()
frame1 = tk.Frame()
frame2 = tk.Frame()

# Add two sliders to the GUI to interactively experiment with filter parameters
slider1 = tk.Scale(frame0, from_=0, to=1)
slider2 = tk.Scale(frame0, from_=0, to=1)
slider1.pack(side=tk.LEFT)
slider2.pack(side=tk.LEFT)

# Create button object used to refresh filtered images with new values
button = tk.Button(frame0, text="Update", command=updateImages)
button.pack(side=tk.BOTTOM)

# Create option menu and create variable to store selected option from options
selectedDrop = tk.StringVar(master)
selectedDrop.set(inspect.getmembers(techniques, inspect.isfunction)[0])
option = tk.OptionMenu(frame0, selectedDrop, list(inspect.getmembers(techniques, inspect.isfunction)))
option.pack(side=tk.TOP)

# Create TK image objects from the pixel arrays as GUI attributes, ready to be displayed in the GUI
master.foetusOriginal = ImageTk.PhotoImage(image=Image.open(FOETUS_PATH_ORIGINAL))
master.nzjersOriginal = ImageTk.PhotoImage(image=Image.open(NZJERS_PATH_ORIGINAL))
try:
    master.foetusFiltered = ImageTk.PhotoImage(image=Image.open(FOETUS_PATH_FILTERED))
    master.nzjersFiltered = ImageTk.PhotoImage(image=Image.open(NZJERS_PATH_FILTERED))
except FileNotFoundError:
    master.foetusFiltered = ImageTk.PhotoImage(image=Image.open(FOETUS_PATH_ORIGINAL))
    master.nzjersFiltered = ImageTk.PhotoImage(image=Image.open(NZJERS_PATH_ORIGINAL))
    pass

# Create labels with four images: two originals and two filtered
labelFoetusOriginal = tk.Label(frame1, image=master.foetusOriginal)
labelNzjersOriginal = tk.Label(frame2, image=master.nzjersOriginal)
labelFoetusFiltered = tk.Label(frame1, image=master.foetusFiltered)
labelNzjersFiltered = tk.Label(frame2, image=master.nzjersFiltered)

# Create reference to image object to account for Tkinter garbage collector bug
labelFoetusOriginal.image = master.foetusOriginal
labelNzjersOriginal.image = master.nzjersOriginal
labelFoetusFiltered.image = master.foetusFiltered
labelNzjersFiltered.image = master.nzjersFiltered

# Pack label images
labelFoetusOriginal.pack(side=tk.LEFT)
labelFoetusFiltered.pack(side=tk.LEFT)
labelNzjersOriginal.pack(side=tk.LEFT)
labelNzjersFiltered.pack(side=tk.LEFT)

# Pack frames
frame0.pack()
frame1.pack(fill=tk.X)
frame2.pack(fill=tk.X)

# Begin GUI loop to enable interaction with GUI window
master.mainloop()




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

# # Think this is broken now
# figures = {'foetus': FOETUS_PATH_ORIGINAL,
#            'NZjers1': NZJERS_PATH_ORIGINAL}
# plotFigs(figures, 2, 1)