""" This program makes the dataset usable for machine learning."""
from __future__ import print_function
import cPickle
import re
from collections import Counter
import numpy as np


PATH_TRAIN = "../train.tsv"
PATH_TEST = "../test.tsv"
COLUMN_LABEL = 5
BATCH_SIZE = 10000  # Don't make this number much larger than 100000
DATA_SIZE = sum(1 for line in open(PATH_TRAIN))  # Number of \n in train file
THRESH = 50  # How often needs a word to occur before it is used

# The following constants are the columns in the "data" variable
COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_SHIPPING = 4
COLUMN_DESCRIPTION = 5

# These characters will be ignored by the neural net when training data
IGN_CHAR = [',', ':', ';', '.', '(', ')', '\'', '"', '!', '?', '*', '&', '^']

# The pickle files
PATH_WORDS = "../words.tsv"
PATH_WORD_MAP = "../word_map.tsv"
PATH_WORD_COUNT = "../word_count.tsv"
PATH_INPUT_VECTOR = "../in_vec.tsv"


def count_words(words):
    """Show plots and statistics on the worduse."""
    count = Counter()
    for _, row in np.ndenumerate(words):
        count.update(row)
    return count


def data_to_input(num_data, words, labels, word_map):
    """Return a version of the data that a neural network can take as input."""
    in_vec = []
    n_words = len(word_map.keys())
    for i, word_row in enumerate(words):
        arr = np.zeros(n_words + len(num_data[0]) + 1)
        for word in word_row:
            if word in word_map:
                arr[word_map[word]] = 1.0
        for j, num in enumerate(num_data[i]):
            arr[-(j + 2)] = num_data[i][j]
        arr[-1] = float(labels[i])
        in_vec.append(arr)
    return np.array(in_vec)


def numeric_data(data):
    """Return an array of floats with for condition and shipping"""
    conditions = data[:,COLUMN_CONDITION]
    shipping = data[:,COLUMN_SHIPPING]
    num_data = np.array([conditions, shipping])
    return num_data.T


def load_data(start, size):
    """Return the trainingdata as an array."""
    data = []
    labels = []
    if start + size > DATA_SIZE:
        size = DATA_SIZE - start

    with open(PATH_TRAIN) as f:
        for _ in range(start):
            f.readline() 
        for _ in range(size):
            line = f.readline()
            line = line[:-1]
            line = line.split("\t")
            label =line[COLUMN_LABEL]
            line = line[1:COLUMN_LABEL] + line[COLUMN_LABEL+1:]
            data.append(np.array(line))
            labels.append(label)

    return np.array(data), np.array(labels)


def map_words(words, count, thresh=1):
    """Gives each word in the array a unique int in a dictionary"""
    word_map = dict()
    n_words = 0
    for _, row in np.ndenumerate(words):
        for _, word in np.ndenumerate(row):
            if word not in word_map.keys() and count[word] > thresh:
                word_map[word] = n_words
                n_words += 1
    return word_map


def sentence_to_words(data):
    """Gives each word in the array a unique int in a dictionary"""
    words = []
    for i in range(data.shape[0]):
        row = data[i]
        string = (row[COLUMN_NAME] + ' ' + row[COLUMN_BRAND] + ' ' + 
                  row[COLUMN_DESCRIPTION])
        string = re.compile(r'[^\s\w_]+').sub(' ', string)  # removes all non-alfanumeric characters
        string = string.lower()
        word_arr = np.array(string.split(' '))
        word_arr = word_arr[word_arr != '']
        words.append(word_arr)
    return np.array(words)


def store(obj, path):
    """Store an object in a file with cPickle."""
    with open(path, 'wb') as out_file:
        cPickle.dump(obj, out_file)


def main():
    """The main function of the program"""
    pointer = 0
    batch_count = 0

    print("Batch size =", BATCH_SIZE, "Data size =", DATA_SIZE)

    data, labels = load_data(0, 1)
    while pointer < DATA_SIZE:
        print("Now starting batch", batch_count, "...")
        new_data, new_labels = load_data(pointer, BATCH_SIZE)
        data = np.concatenate((data, new_data), axis=0)
        labels = np.concatenate((labels, new_labels), axis=0)
        pointer += BATCH_SIZE
        batch_count += 1
    data = data[1:]
    labels = labels[1:]

    words = sentence_to_words(data)
    num_data = numeric_data(data)
    word_count = count_words(words)
    word_map = map_words(words, word_count, thresh=50)
    in_vec = data_to_input(num_data, words, labels, word_map)
    store(in_vec, PATH_INPUT_VECTOR)


main()
