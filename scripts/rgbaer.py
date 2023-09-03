from PIL import Image

from numpy import asarray

image = Image.open('/home/pigi/Downloads/houston_rgb.jpeg')
 
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)


numpydata = asarray(image)
 
# <class 'numpy.ndarray'>
print(type(numpydata))
 
#  shape
print(numpydata.shape)

