#!/usr/bin/env python
import h5py
import numpy
import matplotlib.pyplot as plt
import cv2
import sys
import cv2
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn import datasets, svm, metrics

def view_image(image):

    from matplotlib.pyplot import show, imshow, cm
    imshow(image, cmap=cm.gray)
    show()


#load the presaved model
clf = joblib.load('model.pkl')
#take the image name
nameOfFile =sys.argv[1]

print(nameOfFile)

#convert image into matrix
img = cv2.imread(nameOfFile)
img_ori = img
img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY)
img = cv2.bitwise_not(img) 
img = img.reshape( 1,1,28, 28)


#convert image to array
img = img[0,0,:]

#data normalization used in trainning process
img = img/255.0*2 - 1  

predicted = clf.predict(img.reshape((1,img.shape[0]*img.shape[1] )))
print "Digit on the image predicted is: "
print  predicted
view_image(img_ori )


