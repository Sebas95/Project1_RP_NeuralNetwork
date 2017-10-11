#!/usr/bin/env python
import h5py
import numpy
import matplotlib.pyplot as plt

"""
Train a SVM to categorize 28x28 pixel images into digits (MNIST dataset).
"""

import numpy as np
from sklearn.externals import joblib

def main():
    """Orchestrate the retrival of data, training and testing."""
    data = get_data()

    # Get classifier
    from sklearn.svm import SVC
    clf = SVC(probability=False,  # cache_size=200,
              kernel="rbf", C=2.8, gamma=.0073)

    print("Start fitting. This may take a while")

    # take all of it - make that number lower for experiments
    examples = len(data['train']['X'])
    
    clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])
    joblib.dump(clf, 'model.pkl') 
    

def get_data():
    """
    Get data ready to learn with.

    Returns
    -------
    dict
    """
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    data = {'train': {'X': x_train,'y': y_train}, 'test': {'X': x_test, 'y': y_test}}
    return data

if __name__ == '__main__':
    main()
