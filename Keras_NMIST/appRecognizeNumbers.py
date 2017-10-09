'''
This app read the name of a image and compares it
agains the model and predict the class,printing it in
console
'''

import h5py
import sys
import numpy
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from scipy.misc import imread
from keras.datasets import mnist

#--------------------------TRy to put an image
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
nameOfFile =sys.argv[1]
print(nameOfFile)
#Load de presaved model
model = load_model('my_model.h5')
#Load an image two predict it
test = cv2.imread(nameOfFile)
#Treat it a little two improved classification
test = cv2.cvtColor( test, cv2.COLOR_RGB2GRAY)
test = test.reshape(1, 1, 28, 28)
test = cv2.bitwise_not(test)
#Predict the model class
pred = model.predict_classes(test)
#Print the class predicted
print(pred)
