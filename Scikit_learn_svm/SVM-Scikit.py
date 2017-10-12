from scipy import io
import numpy as np
from sklearn import svm
import scipy.io as sio

#Entrenamiento de la red
X = io.loadmat('xtrain_set.mat')
Y = io.loadmat('ytrain_set.mat')
X = X['X']
Y = Y['Y']
Yaux=  np.zeros((Y.shape[0], 1))
for i in range(0, Y.shape[0]):
	if (Y[i][0]==1):
		Yaux[i]=1
	elif (Y[i][1]==1):
		Yaux[i]=2
	elif (Y[i][2]==1):
		Yaux[i]=3

k_type = 'rbf' #valid poly, sigmoid, linear, rbf, precomputed, 
clf = svm.SVC(gamma=0.0001, C=100, kernel=k_type)
clf.fit(X, Yaux.ravel())  

#prediccion de la red
Xt = io.loadmat('xtest_set.mat')
Xt = Xt['X']
c_matrix = np.zeros((3, 3))
Ypred = clf.predict(Xt)
for j in range(0, Xt.shape[0]):
		ind1= int(Yaux[j]-1)
		ind2= int(Ypred[j]-1)
		c_matrix[ind1][ind2]+=1 

print "using: %d data for training" %Y.shape[0]
print "using: %d data for testing" %Xt.shape[0]
print "kernel_type: %s" %k_type
print "confusion matrix:"
print c_matrix

sensitivity1 = c_matrix[0][0]/(c_matrix[0][0]+c_matrix[0][1]+c_matrix[0][2])
sensitivity2 = c_matrix[1][1]/(c_matrix[1][1]+c_matrix[1][0]+c_matrix[1][2])
sensitivity3 = c_matrix[2][2]/(c_matrix[2][2]+c_matrix[2][0]+c_matrix[2][1])

pres1= c_matrix[0][0]/(c_matrix[0][0]+c_matrix[1][0]+c_matrix[2][0])
pres2 = c_matrix[1][1]/(c_matrix[0][1]+c_matrix[1][1]+c_matrix[2][1])
pres3 = c_matrix[2][2]/(c_matrix[2][2]+c_matrix[0][2]+c_matrix[1][2])

print 'sensitivity 1: {0:.0f}%'.format(sensitivity1*100)
print 'sensitivity 2: {0:.0f}%'.format(sensitivity2*100)
print 'sensitivity 3: {0:.0f}%'.format(sensitivity3*100)

print 'precision 1: {0:.0f}%'.format(pres1*100)
print 'precision 2: {0:.0f}%'.format(pres2*100)
print 'precision 3: {0:.0f}%'.format(pres3*100)
