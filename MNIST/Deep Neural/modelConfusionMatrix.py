'''
 This app compares the expected values and the
 values from the recognition and makes a confusion_matrix
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
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from sklearn.metrics import classification_report

#--------------------Confusion_matrix

model = load_model('my_model.h5')


#Print confusion_matrix
print("The confussion matrix for the data test is")
print "\n"
# Load the original dataset
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
mnist = fetch_mldata('MNIST original')
x = mnist.data
y = mnist.target
# Normalization: Scale data to [-1, 1] - This is of mayor importance!!!
x = x/255.0*2 - 1
x, y = shuffle(x, y, random_state=0)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# reshape to be [samples][pixels][width][height]
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
y_pred = model.predict_classes(X_test) # This will take a few seconds...
y_true = np_utils.to_categorical(y_test)
print confusion_matrix(y_test,y_pred)
print "Accuracy: ", accuracy_score(y_test,y_pred)
print "Classification report: "
print classification_report(y_test,y_pred)
