#!/usr/bin/env python
import h5py
import numpy
import matplotlib.pyplot as plt
import cv2
import sys
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from scipy.misc import *


"""
Train a SVM to categorize 28x28 pixel images into digits (MNIST dataset).
"""

import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn import datasets, svm, metrics

def view_image(image, label=""):
    """
    View a single image.

    Parameters
    ----------
    image : numpy array
        Make sure this is of the shape you want.
    label : str
    """
    from matplotlib.pyplot import show, imshow, cm
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()


#load the presaved model
clf = joblib.load('model.pkl')

#nameOfFile =sys.argv[1]
nameOfFile = 'six28.png'
print(nameOfFile)

img = cv2.imread(nameOfFile)
img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY)
img = cv2.bitwise_not(img)
img = img.reshape( 1,1,28, 28)



img = img[0,0,:]

img = img/255.0*2 - 1  #data normalization

predicted = clf.predict(img.reshape((1,img.shape[0]*img.shape[1] )))
print predicted

