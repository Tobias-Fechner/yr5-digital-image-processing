# Image Enhancement
University of Bath, Year 5

Module has functionality to load test images, apply various filters, and save the filtered images to disk. Used to demonstrate image enhancement techniques taught in the unit Digital Image Processing. 

‼ Cool stuff: abstract base class used to specify common functionality and attributes between all spatial filters, whilst being able to define unique pixel update methodologies for each filter. Likely will implement for Fourier filters but I don't know how yet.

### Filters Implemented: Spatial Filters
* Median
* Mean
* Gaussian
* High pass
* Low pass

### Filters Implemented: Fourier Filters
* Truncate coefficients
