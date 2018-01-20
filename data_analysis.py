"""This program analyses the word use in the data."""
from __future__ import print_function
import cPickle
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords

PATH_TRAIN = "../train.tsv"
PATH_TEST = "../test.tsv"
COLUMN_LABEL = 5
BATCH_SIZE = 100000  # Don't make this number much larger than 100000

# The following constants are the columns in the "data" variable
COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_SHIPPING = 5
COLUMN_DESCRIPTION = 6

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
    # for word, n in count0.most_common(10):
    #     print(word, "occurs", n, "times")

    # A plot a about the distribution of word use.
    #num = len(count0)
    #y = []
    #for _, n in count0.most_common(num):
    #    if n not in y:
    #        y.append(n)
    #plt.plot(y[1:])
    #plt.ylabel("Occurances")
    #plt.show()
    #test


def load_data(start, size):
    """Return the trainingdata as an array."""
    data = []
    stop = False

    with open(PATH_TRAIN) as f:

        i = 0
        for row in f:
            if i < start:
                i += 1
                continue
            elif i >= start + size:
                break

            row = row[:-1]
            row = row.split("\t")
            row = row[1:]
            data.append(np.array(row))
            i += 1

        if i < start + size:
            stop = True

    return stop, np.array(data)


def sentence_to_words(data):
    """Gives each word in the array a unique int in a dictionary"""
    words = []
    for i in range(data.shape[0]):
        row = data[i]
        string = (row[COLUMN_NAME] + ' ' + row[COLUMN_BRAND] + ' ' + 
                  row[COLUMN_DESCRIPTION])
        string = re.compile(r'[^\s\w_]+').sub(' ', string)  # removes all non-alfanumeric characters
        string = string.lower()
        s = set(stopwords.words('english'))
        word_arr = filter(lambda w: not w in s, string.split())
        words.append(word_arr)
    return np.array(words)



def show_counter(word_count, thresh):
    """Count how many words occur at least 'thresh' times."""
    vec_len = 0
    avoid_len = 0
    
    for word in word_count.keys():
        if word_count[word] > thresh:
            vec_len += 1
        else:
            avoid_len += 1

    return vec_len, avoid_len



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


def update_counter(words, word_count):
    for row in words:
        for word in row:
            word_count[word] += 1

    return word_count


def main():
    """The main function of the program."""
    word_count = Counter()

    print("start counting words...")
    pointer = 1
    batch = 1
    stop = False
    while not stop:
        print("... Batch", batch, "currently", show_counter(word_count, 50), "words...")
        stop, data = load_data(pointer, BATCH_SIZE)
        words = sentence_to_words(data)
        word_count = update_counter(words, word_count)
        pointer += BATCH_SIZE
        batch += 1
    print("... Done!", show_counter(word_count, 50), "words.")


main()
