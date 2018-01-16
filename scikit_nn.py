""" This program trains a regressor on the dataset. Running MLPRegressor
requires a sklearn version of 0.18 or higher."""
from __future__ import print_function
from sklearn.neural_network import MLPRegressor
import numpy as np


PATH_TRAIN = "../train.tsv"
PATH_TEST = "../test.tsv"
COLUMN_LABEL = 5
BATCH_SIZE = 100000  # Don't make this number much larger than 100000

# The following constants are the columns in the "data" variable
COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_SHIPPING = 4
COLUMN_DESCRIPTION = 5

# These characters will be ignored by the neural net when training data
IGN_CHAR = [',', ':', ';', '.', '(', ')', '\'', '"', '!', '?', '*', '&', '^']


def load_data():
    """Return the trainingdata as an array"""
    data = []
    labels = []
    with open(PATH_TRAIN) as f:
        f.readline()
        for _ in range(BATCH_SIZE):
            line = f.readline()
            line = line[:-1]
            line = line.split("\t")
            label = float(line[COLUMN_LABEL])
            line = line[1:COLUMN_LABEL] + line[COLUMN_LABEL+1:]
            data.append(np.array(line))
            labels.append(label)
    return np.array(data), np.array(labels)


def map_words(start_int, word_array):
    """Gives each word in the array a unique int in a dictionary"""
    word_map = dict()
    n_words = 0
    int_array = word_array
    for i in range(word_array.shape[0]):
        string = word_array[i]
        string.strip()
        string.lower()
        for char in IGN_CHAR:
            string.replace(char, '')

        for word in string.split(' '):
            if word not in word_map.keys():
                word_map[word] = n_words + start_int
                n_words += 1

    return n_words, word_map


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
    sq_e = 0
    n = labels.shape[0]
    for i in range(n):
        pred = regr.predict(data[i])
        sq_e += np.power(pred - labels[i], 2)
    return sq_e / n



def train_regressor(data, labels):
    """Return a regression neural network that is fitted to the data"""
    regr = MLPRegressor()
    data = list(data)
    for i in range(len(data)):
        data[i] = list(data[i])
    regr.fit(data, list(labels))
    return regr


def main():
    """The main function of the program"""
    data, labels = load_data()
    print("step1")
    n_words, name_map = map_words(0, data[:,COLUMN_NAME])
    print("step2")
    n_words, brand_map = map_words(n_words, data[:,COLUMN_BRAND])
    print("step3")
    desc_map = map_words(n_words, data[:,COLUMN_DESCRIPTION])
    
    train_data, train_labels, val_data, val_labels = split_train_val(data, labels)
    # regr = train_regressor(train_data, train_labels)
    # sq_e = test_regressor(regr, val_data, val_labels)
    # print("The regression has an average squared error of:", sq_e)


main()
