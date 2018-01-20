""" This program trains a regressor on the dataset. Running MLPRegressor
requires a sklearn version of 0.18 or higher.
    This program expects a file that cPickle can open at location PATH_TRAIN
cPickle needs to load a 2 dimensional numpy array, each row is a datapoint and
the last element of each row should be the label."""
from __future__ import print_function
import cPickle
from sklearn.neural_network import MLPRegressor
import numpy as np


PATH_TRAIN = "../input_train_50.tsv"
PATH_VALIDATION = "../input_val_50.tsv"
PATH_SPEC = "../input_specs_50.tsv"

BATCH_SIZE = 10000  # Don't make this much bigger than 10,000
TRAIN_SIZE = sum(1 for line in open(PATH_TRAIN))
VAL_SIZE = sum(1 for line in open(PATH_TRAIN))
VALIDATION_SIZE = sum(1 for line in open(PATH_VALIDATION))

MAX_TRIES = 200
CONVERGENCE = 0.01


def load_train(path, start, size):
    """Return the trainingdata as an array"""
    words_len, cats_len = load_specs(PATH_SPEC)
    vec_len = words_len + cats_len + 2
    in_vecs = []
    labels = []
    with open(path, 'rb') as in_file:

        for i, l in enumerate(in_file):
            if i < start:
                continue
            elif i >= start + size:
                break

            vec = np.zeros(vec_len)
            l = l.split(',')

            words = l[0].split('\t')
            for word in words:
                if word == '': # Sometimes a word is empty.
                    continue
                vec[int(word)] = 1.0

            cats = l[1].split('\t')
            for cat in cats:
                if cat == '':  # Sometimes a category is empty.
                    break
                vec[int(cat) + words_len] = 1.0

            nums = l[-3:-1]
            for j, num in enumerate(nums):
                vec[j + words_len + cats_len] = float(num)
            label = float(l[-1][:-1])

            in_vecs.append(vec)
            labels.append(label)
    return np.array(in_vecs), np.array(labels)


def load_specs(path):
    """Return the number of words and number of categories in the file."""
    words_len = 0
    cats_len = 0
    with open(path) as f:
        l = f.readline()
        l = l.split('\t')
        words_len = int(l[0])
        cats_len = int(l[1])
    return words_len, cats_len


def load_val(path, batch_size, file_size):
    """Return some random rows from the validation file."""
    if batch_size > file_size:
        batch_size = file_size
    
    indexes = np.arange(file_size)
    np.random.shuffle(indexes)
    indexes = set(indexes[:batch_size])
    words_len, cats_len = load_specs(PATH_SPEC)
    vec_len = words_len + cats_len + 2

    val_data = []
    val_labels = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i not in indexes:
                continue

            vec = np.zeros(vec_len)
            l = l.split(',')

            words = l[0].split('\t')
            for word in words:
                if word == '': # Sometimes a word is empty.
                    continue
                vec[int(word)] = 1.0

            cats = l[1].split('\t')
            for cat in cats:
                if cat == '':  # Sometimes a category is empty.
                    break
                vec[int(cat) + words_len] = 1.0

            nums = l[-3:-1]
            for j, num in enumerate(nums):
                vec[j + words_len + cats_len] = float(num)

            label = float(l[-1][:-1])

            val_data.append(vec)
            val_labels.append(label)
    return np.array(val_data), np.array(val_labels)


def test_regressor(regr, data, labels):
    """Return the average squared error of the regression neural network"""
    pred = regr.predict(data)
    return np.sum(np.power(pred - labels, 2)) / len(labels)


def main():
    """The main function of the program"""
    num_batches = int(TRAIN_SIZE / BATCH_SIZE)
    print("The program will run in", num_batches)

    past_err = long(999999)
    new_err = 0
    regr = MLPRegressor()

    for _ in range(MAX_TRIES):

        for batch in range(num_batches):
            print("\nStarting batch", batch + 1, "of", num_batches)

            print("Loading data...")
            train_data, train_labels = load_train(PATH_TRAIN, batch * BATCH_SIZE, BATCH_SIZE)

            print("Training neural network...")
            regr.partial_fit(train_data, train_labels)

        print("\nTesting the network...")
        val_data, val_labels = load_val(PATH_VALIDATION, BATCH_SIZE, VAL_SIZE)
        new_err = test_regressor(regr, val_data, val_labels)

        if past_err - new_err < CONVERGENCE:
            print("Convergence has reached:", past_err - new_err)
            break

        print("Current error:", np.sqrt(new_err), "improvement",
                                    np.sqrt(past_err) - np.sqrt(new_err), "\n")
        past_err = new_err

    print("Done!\nThe regression has an average error of:", np.sqrt(new_err))


main()
