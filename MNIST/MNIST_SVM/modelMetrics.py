from sklearn.externals import joblib
import h5py
import numpy
from sklearn.svm import SVC

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

def analyze(clf, data):
    """
    Analyze how well a classifier performs on data.

    Parameters
    ----------
    clf : classifier object
    data : dict
    """
    # Get confusion matrix
    from sklearn import metrics
    predicted = clf.predict(data['test']['X'])
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(data['test']['y'],
                                   predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(data['test']['y'], predicted))
    print("Precision score: %0.4f" % metrics.precision_score(data['test']['y'], predicted))
    print("Precision score: %0.4f" % metrics.classification_report(data['test']['y'], predicted))
    

data = get_data()
clf = joblib.load('model.pkl')
analyze(clf, data)
