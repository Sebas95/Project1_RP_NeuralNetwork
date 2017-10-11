'''
This app read the name of a image and compares it
agains the model and predict the class,printing it in
console
'''

from sklearn.metrics import classification_report, confusion_matrix
import h5py
import sys
import numpy
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
from scipy.misc import imread
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils



#--------------------------TRy to put an image
nameOfFile =sys.argv[1]
print(nameOfFile)
#Load de presaved model
model = load_model('my_model.h5')
#Load an image two predict it
img = cv2.imread(nameOfFile)
#Treat it a little two improved classification
img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY)
img = img.reshape(1, 1, 28, 28)
img = cv2.bitwise_not(img)

#This make an array reshape with the first element, the quantity of elements
#wanted and the last two the size in with and heigth of the image

#Predict the model class
#pred = model.predict_classes(X_test)
pred = model.predict_classes(img)
#Print the class predicted
print "------------------------------------"
print "-------------Result-----------------"
print "------------------------------------"
print "The number in the image is: ",pred[0]
