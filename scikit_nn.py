""" This program trains a regressor on the dataset. Running MLPRegressor
requires a sklearn version of 0.18 or higher.
    This program expects a file that cPickle can open at location PATH_INPUT
cPickle needs to load a 2 dimensional numpy array, each row is a datapoint and
the last element of each row should be the label."""
from __future__ import print_function
import cPickle
from sklearn.neural_network import MLPRegressor
import numpy as np


PATH_INPUT = "../in_vec"


def load_data():
    """Return the trainingdata as an array"""
    in_vec = []
    with open(PATH_INPUT, 'rb') as in_file:
        in_vec = cPickle.load(in_file)
    return in_vec


def split_labels(in_vec):
    """Return the data for the nn and the label seperate from eachother"""
    return in_vec[:, :-1], in_vec[:, -1]


def split_train_val(data, labels, size=0.7):
    """Returns a training and a validation array"""
    i = int(labels.shape[0] * size)
    rand_data = np.random.permutation(data)
    train_data = rand_data[:i]
    val_data = rand_data[i:]

    rand_labels = np.random.permutation(labels)
    train_labels = rand_labels[:i]
    val_labels = rand_labels[i:]
    return train_data, train_labels, val_data, val_labels


def test_regressor(regr, data, labels):
    """Return the average squared error of the regression neural network"""
    pred = regr.predict(data)
    return np.sum(np.power(pred - labels, 2))


def train_regressor(data, labels):
    """Return a regression neural network that is fitted to the data"""
    regr = MLPRegressor()
    regr.fit(data, labels)
    return regr


def main():
    """The main function of the program"""
    print("Loading data...")
    input_data = load_data()
    data, labels = split_labels(input_data)
    train_data, train_labels, val_data, val_labels = split_train_val(data, labels)

    print("Done!\nTraining neural network...")
    regr = train_regressor(train_data, train_labels)
    print("Done!\nTesting the network...")
    sq_e = test_regressor(regr, val_data, val_labels)
    print("Done!\nThe regression has an average error of:", np.sqrt(sq_e))


main()
