from scipy import io
import numpy as np
from sklearn import svm
import scipy.io as sio

#Entrenamiento de la red
X = io.loadmat('xtrain_set.mat')
Y = io.loadmat('ytrain_set.mat')
X = X['X']
Y = Y['Yreal']
Yaux=  np.zeros((Y.shape[0], 1))
for i in range(0, Y.shape[0]):
	if (Y[i][0]==1):
		Yaux[i]=1
	elif (Y[i][1]==1):
		Yaux[i]=2
	elif (Y[i][2]==1):
		Yaux[i]=3

clf = svm.SVC(gamma=0.0001, C=100)
clf.fit(X, Yaux.ravel())  

#prediccion de la red
Xt = io.loadmat('xtest_set.mat')
Xt = Xt['X']
c_matrix = np.zeros((3, 3))
Ypred = clf.predict(X)
for j in range(0, Xt.shape[0]):
	if (Ypred[j]==1):
		if (Yaux[j]==1):
			c_matrix[0][0]+=1 
		elif (Yaux[j]==2):
			c_matrix[1][0]+=1
		elif (Yaux[j]==3):
			c_matrix[2][0]+=1
	elif (Ypred[j]==2):
		if (Yaux[j]==1):
			c_matrix[0][1]+=1
		elif (Yaux[j]==2):
			c_matrix[1][1]+=1
		elif (Yaux[j]==3):
			c_matrix[2][1]+=1
	elif (Ypred[j]==3):
		if (Yaux[j]==1):
			c_matrix[0][2]+=1
		elif (Yaux[j]==2):
			c_matrix[1][2]+=1
		elif (Yaux[j]==3):
			c_matrix[2][2]+=1

print c_matrix

