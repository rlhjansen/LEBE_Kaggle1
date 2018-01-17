import numpy as np
import re
from collections import Counter

PATH_TRAIN = "../train.tsv"
PATH_TEST = "../test.tsv"
COLUMN_LABEL = 5
BATCH_SIZE = 10000  # Don't make this number much larger than 100000
DATA_SIZE = sum(1 for line in open(PATH_TRAIN))  # Number of \n in train file
THRESH = 50  # How often needs a word to occur before it is used


COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_SHIPPING = 4
COLUMN_DESCRIPTION = 5


# This is a stub file, a file in which all (important) functions are explained.


def map_words(word_map, words, count, thresh=1):
    """Gives each word in the array a unique int in a dictionary"""
    n_words = 0
    for _, row in np.ndenumerate(words):
        for _, word in np.ndenumerate(row):
            if word not in word_map.keys() and count[word] > thresh:
                word_map[word] = n_words
                n_words += 1
    return word_map


def map_cats(cats, count, thresh=1, startval=0):
    cat_map = dict()
    n_words = startval
    for _, row in np.ndenumerate(cats):
        for _, cat in np.ndenumerate(row):
            if cat not in cat_map.keys() and count[cat] > thresh:
                cat_map[cat] = n_words
                n_words += 1
    return cat_map

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


def count_words(words, count):
    """Show plots and statistics on the worduse."""
    for _, row in np.ndenumerate(words):
        count.update(row)
    return count

def count_cats(cats, count):
    for _, row in np.ndenumerate(cats):
        count.update(row)
    return count


def numeric_data(data):
    """Return an array of floats for condition and shipping"""
    conditions = data[:, COLUMN_CONDITION]
    shipping = data[:, COLUMN_SHIPPING]
    num_data = np.array([conditions, shipping])
    return num_data.T

def row_to_cats(data):
    cats = []
    for i in range(data.shape[0]):
        row = data[i]
        category_strings = row[COLUMN_CATEGORY].split('/')
        for j, string in enumerate(category_strings):
            string = re.compile(r'[^\s\w_]+').sub(' ', string)  # removes all non-alfanumeric characters
            category_strings[j] = string.lower()
        category_strings = [i for i in category_strings if i != '']
        cats.append(category_strings)
    return np.array(cats)


# num_data: n x 2 float array; 1e colom condition, 2e colom shipping
# words: n x 1 string array; elke rij: woorden die voorkomen in die rij
# labels: n x 1 float array: prijzen van de rij
# word_map: dict() die woorden mapt met ints
def data_to_input(num_data, words, labels, word_map, cat_map):
    """Return a version of the data that a neural network can take as input."""
    in_vecs = []
    n_words = len(word_map.keys())
    for i, word_row in enumerate(words):
        in_row = np.zeros(n_words + len(num_data[0]) + 1)  # This becomes a row for "in_vecs"
        for word in word_row:
            if word in word_map:
                in_row[word_map[word]] = 1.0
        for j, _ in enumerate(num_data[i]):
            in_row[-(j + 2)] = num_data[i][j] 
        in_row[-1] = float(labels[i])
        in_vecs.append(in_row)
    return np.array(in_vecs)
# in_vecs: [[word word word ... word shipping condition price] ... ]
#Todo add categorical to above

# start: int; How many lines to skip before reading the data file.
# size: int; How many lines to read.
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
            line = line[:-1]  # Remove newline
            line = line.split("\t")
            label = line[COLUMN_LABEL]
            line = line[1:COLUMN_LABEL] + line[COLUMN_LABEL+1:]
            data.append(np.array(line))
            labels.append(label)

    return np.array(data), np.array(labels)
# data: n x 6 array; 
# label: n x 1 array; prices


# start: int; How many lines to skip before reading the data file.
# size: int; How many lines to read.
def load_input(start, size):
    """Return input vectors as an 2-dim array."""
    pass
# data
# label: n x 1 array; prices


# This is the main function of the preprocessing.py
def main():
    """The main function of the program"""
    batch_count = 0
    pointer = 0
    continues = True
    word_counter = Counter()
    cat_counter = Counter()
    print("Batch size =", BATCH_SIZE, "Data size =", DATA_SIZE)

    data, labels = load_data(pointer, BATCH_SIZE)
    PATH_INPUT_VECTOR = derive_filename()
    while continues:
        data = data[1:]
        labels = labels[1:]

        words = sentence_to_words(data)  # process name, brand, description
        cats = row_to_cats(data)
        cat_counter = count_cats(cats, cat_counter) # return a counter of all categories
        num_data = numeric_data(data)  # process condition, shipping
        word_counter = count_words(words, word_counter)  # return a counter of all words
        word_map = map_words(words, word_counter)  # return a map[word] = int
        category_map = map_cats(cats, cat_counter)
        data, labels, continues = load_data(pointer, BATCH_SIZE)
        print("Now indexed batch", batch_count, "...")
        pointer += BATCH_SIZE
        batch_count += 1

    continues = True
    while continues:
        in_vecs = data_to_input(num_data, words, word_map, cats, category_map, labels)
        PATH_INPUT_VECTOR = derive_filename()
        store(in_vecs, PATH_INPUT_VECTOR, pointer)  # ToDo
        data, labels, continues = load_data(pointer, BATCH_SIZE)

        print("Now starting batch", batch_count, "...")
        pointer += BATCH_SIZE
        batch_count += 1
        print("done with batch", batch_count)


# returns string with path for specific run values
# retval = "../invec_tresh50.tsv
def derive_filename():
    base = "../invec_"
    threshstring = "tresh"+str(THRESH)
    end = ".tsv"
    retval = base+threshstring+end
    return retval



# arr: any 2-dim array
# path: string; path to storage file
# pointer: int; from which byte in the file do you start storing the "arr"
def store(arr, path, pointer):
    with open(path, "a") as vec_file:
        for row in arr:
            string = "\t".join([str(row[i]) for i in range(row.shape[0])])
            vec_file.write(string)
    vec_file.close()

# pointer: int; at which byte did you stop storing
# how to store: word word word ... word shipping condition price\n ... (use tabs instead of spaced)


main()