""" This program makes the dataset usable for machine learning."""
from __future__ import print_function
import re
from collections import Counter
import numpy as np
from nltk.corpus import stopwords

PATH_TRAIN = "../train.tsv"
PATH_TEST = "../test.tsv"
BATCH_SIZE = 100000  # Don't make this number much larger than 100000
DATA_SIZE = sum(1 for line in open(PATH_TRAIN))  # Number of \n in train file
THRESH = 50  # How often needs a word to occur before it is used
TRAIN_RATIO = 0.8  # The ratio train validation

# The following constants are the columns in each "data" variable
COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_PRICE = 4
COLUMN_SHIPPING = 5
COLUMN_DESCRIPTION = 6

# These characters will be ignored by the neural net when training data
IGN_CHAR = [',', ':', ';', '.', '(', ')', '\'', '"', '!', '?', '*', '&', '^']

# The output files
PATH_INPUT_TRAIN = "../input_train_" + str(THRESH) + ".tsv"
PATH_INPUT_VAL = "../input_val_" + str(THRESH) + ".tsv"
PATH_INPUT_SPECS = "../input_specs_" + str(THRESH) + ".tsv"


def categorical_data(data):
    """Return a vectorized form of the category column"""
    cats = []
    for i in range(data.shape[0]):
        row = data[i]
        string = row[COLUMN_CATEGORY]
        string = re.compile(r'[^\s\w_]+').sub(' ', string)  # removes all non-alfanumeric characters
        string = string.lower()
        cat_arr = np.array(string.split(' '))
        cat_arr = filter_words(cat_arr)
        cats.append(cat_arr)
    return np.array(cats)


def count_words(words, word_count):
    """Show plots and statistics on the worduse."""
    for _, row in np.ndenumerate(words):
        word_count.update(row)
    return word_count


def data_to_input(num_data, words, cats, labels, word_map, cat_map):
    """Return a version of the data that a neural network can take as input."""
    in_vec = []
    for i, word_row in enumerate(words):
        vec = ""

        for word in word_row:
            if word in word_map:
                vec += str(word_map[word]) + '\t'
        vec = vec[:-1]
        vec += ','

        for cat in cats[i]:
            if cat in cat_map:
                vec += str(cat_map[cat]) + '\t'
        vec = vec[:-1]
        vec += ','

        for j, _ in enumerate(num_data[i]):
            vec += str(num_data[i][j]) + ','
        vec += labels[i]

        in_vec.append(vec + '\n')
    return np.array(in_vec)


def filter_words(word_arr):
    """Takes a array of words and removes some of them, then returns the
    smaller array."""
    word_arr = word_arr[word_arr != '']
    # This is a very ugly loop, if the nltk works, please remove it.
    for word in word_arr:
        if len(word) < 3:  # I chose 3 as an arbitrary length
            word_arr = word_arr[word_arr != word]
    return word_arr


def labels_from_data(data):
    """Return an array with the prizes matching the rows"""
    labels = data[:, COLUMN_PRICE]
    return labels


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


def make_maps():
    """Create a word map that assigns unique integers to words."""
    batch = 1
    pointer = 1
    word_map = dict()
    word_count = Counter()
    cat_map = dict()
    cat_count = Counter()

    print("Making a wordmap:")
    stop = False
    while not stop:
        print("\nNow loading batch", batch, "...")
        stop, data = load_data(pointer, BATCH_SIZE)
        print("... Substracting the words ...")
        words = words_from_data(data)
        cats = categorical_data(data)
        print("... Counting words ...")
        word_count = count_words(words, word_count)
        cat_count = count_words(cats, cat_count)
        print("... Mapping words ...")
        word_map = map_words(word_count, word_map)
        cat_map = map_words(cat_count, cat_map)
        print("... Currently", len(word_map.keys()), "words are in the map...")
        print("... And", len(cat_map.keys()), "categories are in the map.")
        batch += 1
        pointer += BATCH_SIZE

    return word_map, cat_map


def map_words(word_count, word_map, thresh=THRESH):
    """Gives each word in the array a unique int in a dictionary"""
    n_words = len(word_map.keys())
    for word in word_count.keys():
        if word_count[word] >= thresh and not word_map.has_key(word):
            word_map[word] = n_words
            n_words += 1
    return word_map


def numeric_data(data):
    """Return an array of floats for condition and shipping"""
    conditions = data[:, COLUMN_CONDITION]
    shipping = data[:, COLUMN_SHIPPING]
    num_data = np.array([conditions, shipping])
    return num_data.T


def split_train_val(data, ratio):
    """Return a set for training and a set for validation"""
    np.random.shuffle(data)
    index = int(len(data) * ratio)
    train = data[:index]
    val = data[index:]
    return train, val


def store(arr, path):
    """Store an array in a file."""
    with open(path, "a") as vec_file:
        for row in arr:
            vec_file.write(row)


def store_specs(path, word_len, cat_len):
    """Store important information about the train and val files."""
    with open(path, 'w') as f:
        f.write(str(word_len) + '\t' + str(cat_len))


def words_from_data(data):
    """Gives each word in the array a unique int in a dictionary"""
    words = []
    for i in range(data.shape[0]):
        row = data[i]
        string = (row[COLUMN_NAME] + ' ' + row[COLUMN_BRAND] + ' ' +
                  row[COLUMN_DESCRIPTION])
        string = re.compile(r'[^\s\w_]+').sub(' ', string)  # removes all non-alfanumeric characters
        string = string.lower()
        word_arr = np.array(string.split(' '))
        word_arr = filter_words(word_arr)
        words.append(word_arr)
    return np.array(words)


def main():
    """The main function of the program"""
    word_map, cat_map = make_maps()
    store_specs(PATH_INPUT_SPECS, len(word_map), len(cat_map))
    batch = 1
    pointer = 1

    print("\n\nStoring the input data")
    print("Batch size =", BATCH_SIZE, "Data size =", DATA_SIZE)

    stop = False
    while not stop:
        print("\nNow loading batch", batch, "...")
        stop, data = load_data(pointer, BATCH_SIZE)
        train, val = split_train_val(data, TRAIN_RATIO)

        print("... converting to input ...")
        num_train = numeric_data(train)
        num_val = numeric_data(val)
        words_train = words_from_data(train)
        words_val = words_from_data(val)
        cats_train = categorical_data(train)
        cats_val = categorical_data(val)
        labels_train = labels_from_data(train)
        labels_val = labels_from_data(val)
        in_train = data_to_input(num_train, words_train, cats_train,
                                 labels_train, word_map, cat_map)
        in_val = data_to_input(num_val, words_val, cats_val,
                               labels_val, word_map, cat_map)

        print("... storing")
        store(in_train, PATH_INPUT_TRAIN)
        store(in_val, PATH_INPUT_VAL)

        batch += 1
        pointer += BATCH_SIZE


main()
