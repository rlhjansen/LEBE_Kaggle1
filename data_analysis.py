"""This program analyses the word use in the data."""
from __future__ import print_function
import cPickle
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

PATH_TRAIN = "../train.tsv"
PATH_TEST = "../test.tsv"
PATH_WORD_FILE = "../words.tsv"
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


def count_words(words):
    """Show plots and statistics on the worduse."""
    count0 = Counter()
    for _, row in np.ndenumerate(words):
        count0.update(row)

    # We also want to count how many words only occur n times.
    count1 = Counter(count0.values())

    # How long would the vector matrix be if words that occur <= than n_times
    num = len(count0.keys())
    print(num)
    for n in range(1, 51):
        num -= count1[n]
        print("The vector would be", num, "long for n value:", n)

    # The most common words
    for word, n in count0.most_common(10):
        print(word, "occurs", n, "times")

    # A plot a about the distribution of word use.
    num = len(count0)
    y = []
    for _, n in count0.most_common(num):
        if n not in y:
            y.append(n)
    plt.plot(y[1:])
    plt.ylabel("Occurances")
    plt.show()


def load_data():
    """Return the trainingdata as an array."""
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


def string_occurance(string, words):
    """Returns the indexes in which rows a string occurs."""
    indexes = []
    for i, row in np.ndenumerate(words):
        if string in row:
            indexes.append(i[0])
    return np.array(indexes)


def main():
    """The main function of the program."""
    data, labels = load_data()
    words = sentence_to_words(data)
    # store(words, PATH_WORD_FILE)
    # count_words(words)
    indexes = string_occurance('xl', words)
    print(data[indexes])


main()
