"""This program will calculate which words are used very different between
classes."""
from __future__ import print_function, division
from pathlib import Path
import re
from collections import Counter
import numpy as np
from nltk.corpus import stopwords

# Important parameters on how the program is run.
BATCH_SIZE = 100000  # Don't make this much larger than 100000
THRESH = 50
VEC_LEN = 100

# The input files
PATH_TRAIN = "../train.tsv"
PATH_CLASSES = "../split_classes/"

# The output file
PATH_OUT = "../delta_vec_" + str(VEC_LEN) + ".tsv"

# The following constants are the columns in each "data" variable
COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_PRICE = 4
COLUMN_SHIPPING = 5
COLUMN_DESCRIPTION = 6


def count_data(path):
    """Returns a dictionary with a word matched with it's relative occurance"""
    pointer = 0
    batch = 1
    word_count = Counter()
    tot_words = 0

    print("Start counting all words in the data set.")
    stop = False
    while not stop:
        print("Now counting batch", batch, "...")
        stop, data = load_data(path, pointer, BATCH_SIZE)
        words = words_from_data(data)
        num_words, word_count = update_counter(words, word_count)
        tot_words += num_words

        pointer += BATCH_SIZE
        batch += 1
    
    word_occ = normalize_counter(tot_words, word_count)
    return word_occ


def count_words(path_obj):
    """Returns a dictionary with a word matched with it's relative occurance"""
    pointer = 0
    word_count = Counter()
    tot_words = 0

    stop = False
    while not stop:
        stop, words, labels = load_words(path_obj, pointer, BATCH_SIZE)
        num_words, word_count = update_counter(words, word_count)
        tot_words += num_words

        pointer += BATCH_SIZE
    
    word_occ = normalize_counter(tot_words, word_count)
    return word_occ, labels


def get_class_name(path):
    """Return the name of the class from a path string."""
    class_name = path.split('/')[-1]
    return class_name[:-4]


def init_delta_occ(word_occ):
    """Return a word map with all words match with a 0.0"""
    delta_occ = dict()
    for key in word_occ.keys():
        delta_occ[key] = 0.0
    return delta_occ


def load_data(path, start, size):
    """Return the trainingdata as an array."""
    data = []
    stop = False

    with open(path) as f:

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


def load_words(path_obj, start, size):
    """Return the words and labels from a file. Also return if the end of the 
    file is reached."""
    words = []
    labels = []
    stop = False

    with path_obj.open() as f:

        i = 0
        for row in f:
            if i < start:
                i += 1
                continue
            elif i >= start + size:
                break

            row = row[:-1]
            row = row.split("\t")
            s = set(stopwords.words('english'))
            word_arr = np.array(filter(lambda w: not w in s, row[0].split(' ')))
            strlen = np.vectorize(len)
            word_arr = word_arr[strlen(word_arr) > 1]
            words.append(word_arr)
            labels.append(float(row[1]))
            i += 1

        if i < start + size:
            stop = True

    return stop, np.array(words), np.array(labels)


def normalize_counter(num_words, word_count):
    """Return a dictionary that maps a word with its relative use."""
    word_occ = dict()
    for word in word_count.keys():
        word_occ[word] = word_count[word] / num_words
    return word_occ


def store_vec(vec):
    """Store the delta vector in a file."""
    with open(PATH_OUT, 'w') as f:
        for word in vec:
            f.write(word + '\n')


def update_counter(words, word_count):
    """Count word occurance and count the total amount of words."""
    num = 0
    for row in words:
        for word in row:
            word_count[word] += 1
            num += 1
    return num, word_count


def vec_occ(vec):
    """Return how many rows have a word corresponding with the vector."""
    pointer = 1
    batch = 1
    num_rows = 0
    tot_rows = 0

    stop = False
    while not stop:
        print("Now checking for", batch, "current =", num_rows / (tot_rows + 1) , "...")
        stop, data = load_data(PATH_TRAIN, pointer, BATCH_SIZE)
        words = words_from_data(data)
        for row in words:
            tot_rows += 1
            for word in row:
                if word in vec:
                    num_rows += 1
                    break

        pointer += BATCH_SIZE
        batch += 1
    
    return num_rows / tot_rows


def words_from_data(data):
    """Return an array of words that occur in the name-, brand-, and
    description column of the data set."""
    strings = []
    for i in range(data.shape[0]):
        row = data[i]
        string = (row[COLUMN_NAME] + ' ' + row[COLUMN_BRAND] + ' ' +
                  row[COLUMN_DESCRIPTION])
        string = re.compile(r'[^\s\w_]+').sub(' ', string)  # removes all non-alfanumeric characters
        string = string.lower()
        s = set(stopwords.words('english'))
        word_arr = np.array(filter(lambda w: not w in s, string.split(' ')))
        strlen = np.vectorize(len)
        word_arr = word_arr[strlen(word_arr) > 1]
        strings.append(word_arr)
    return np.array(strings)


def main():
    """The main function of the program"""
    tot_word_occ = count_data(PATH_TRAIN)
    delta_occ = init_delta_occ(tot_word_occ)

    pathlist = Path(PATH_CLASSES)
    print("\nStart comparing words with the classes.")

    for path_obj in pathlist.iterdir():
        class_name = get_class_name(str(path_obj))
        print("Comparing words from \"" + class_name + "\"")
        word_occ, class_labels = count_words(path_obj)
        for word in word_occ:
            delta_occ[word] += abs(word_occ[word] - tot_word_occ[word])

    print("\nBiggest delta words:")
    max_deltas = sorted(delta_occ, key=delta_occ.__getitem__, reverse=True)
    max_deltas = max_deltas[:VEC_LEN]
    for word in max_deltas:
        print(word, delta_occ[word])

    store_vec(max_deltas)

    print("\nCalculating vector occurance")
    print(vec_occ(max_deltas))


main()
