#!/usr/bin/env python
import h5py
import numpy
import matplotlib.pyplot as plt
import cv2
import sys


"""
Train a SVM to categorize 28x28 pixel images into digits (MNIST dataset).
"""

import numpy as np
from sklearn.externals import joblib



#load the presaved model
clf = joblib.load('model.pkl')

nameOfFile =sys.argv[1]
print(nameOfFile)
#Load an image two predict it
test = cv2.imread(nameOfFile)
#Treat it a little two improved classification
test = cv2.cvtColor( test, cv2.COLOR_RGB2GRAY)
test = test.reshape(1, 1, 28, 28)
test = cv2.bitwise_not(test)
#Predict the model class
predicted = clf.predict(data['test']['X'])
#predicted = clf.predict(test)
#Print the class predicted
print(predicted)
