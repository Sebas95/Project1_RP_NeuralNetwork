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
from keras.utils import np_utils

#--------------------Confusion_matrix

model = load_model('my_model.h5')

#List of all values for the test
namesList = ["six28.png","two28.png","sev28.png","zero28.png","zero228.png","sev228.png","three28.png","five28.png", "four28.png","nine28.png","zero328.png","eigth28.png"]

true = [8,0,9,4,5,3,7,0,0,7,2,6];
'''
#false = [0,4,5,6]
#print confusion_matrix(true,false);
'''


#For to get all image in the list of  names for image test

contador = 0
for value in namesList:
    newImg = cv2.imread(value)
    #Treat it a little two improved classification
    newImg = cv2.cvtColor( newImg, cv2.COLOR_RGB2GRAY)
    newImg = newImg.reshape(1, 1, 28, 28)
    newImg = cv2.bitwise_not(newImg)
    if(contador == 0):
        arrayImages = np.array(newImg)
    else:
        arrayImages = np.append(newImg,arrayImages)
    contador = contador + 1

# reshape to be [samples][pixels][width][height]
arrayImages = arrayImages.reshape(len(true),1,28,28).astype('float32')

#print arrayImages
print "\n"
print "The predicted values are: "
print model.predict_classes(arrayImages).tolist()
print "The true values are"
print true
#Print confusion_matrix
print("The confussion matrix for the data test is")
print "\n"
print confusion_matrix(true,model.predict_classes(arrayImages))
