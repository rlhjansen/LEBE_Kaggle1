"""This program will calculate which words are used very different between
classes."""
from __future__ import print_function
import re
from collections import Counter
import numpy as np

# The input file
PATH_TRAIN = "../train.tsv"

# The pre- and postfix on all output files
PATH_PRE = "../split_classes/"
PATH_POST = ".tsv"

# Important parameters on how the program is run.
BATCH_SIZE = 100000  # Don't make this much larger than 100000
THRESH = 50
MAX_DEPTH = 2  # We don't take very niche subcategories.

# The following constants are the columns in each "data" variable
COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_PRICE = 4
COLUMN_SHIPPING = 5
COLUMN_DESCRIPTION = 6

# These constants are the possible values found in the column
POS_SHIPPING = np.array([0.0, 1.0])
POS_CONDITION = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def get_classes(data):
    """Return an array over all columns which will be split on."""
    conditions = data[:, COLUMN_CONDITION]
    categories = data[:, COLUMN_CATEGORY]
    shipping = data[:, COLUMN_SHIPPING]
    return np.array([conditions, categories, shipping]).T


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


def name_class(arr):
    """Return the values of the class in a string"""
    depth = len(arr[1])
    if depth > MAX_DEPTH:
        depth = MAX_DEPTH
    
    cats = arr[1].split('/')
    arr[1] = ""   
    for i in range(depth):
        arr[1] += cats[i] + '-'
    arr[1] = arr[1][:-1]

    name = ""
    for elem in arr:
        name += str(elem) + '_'

    return name[:-1]


def possible_cats(path):
    """Return a set of possible categories in the data."""
    pointer = 0
    batch = 1
    cats = set()

    print("Start getting all categories in the data set.")
    stop = False
    while not stop:
        print("Now getting categories from batch", batch, "...")
        stop, data = load_data(pointer, BATCH_SIZE)
        cats.update(data[:, COLUMN_CATEGORY])

        pointer += BATCH_SIZE
        batch += 1
    
    return cats


def store_classes(classes, strings, labels):
    """Take a file named after the class, and store the strings + labels in 
    them, separated by a tab."""
    for i, _ in enumerate(labels):
        class_name = name_class(classes[i])
        string = strings[i]
        label = labels[i]
        with open(PATH_PRE + class_name + PATH_POST, 'a') as f:
            f.write(string + '\t' + label + '\n')


def strings_from_data(data):
    """Return an array of strings that occur in the name-, brand-, and 
    description column of the data set."""
    strings = []
    for i in range(data.shape[0]):
        row = data[i]
        string = (row[COLUMN_NAME] + ' ' + row[COLUMN_BRAND] + ' ' +
                  row[COLUMN_DESCRIPTION])
        string = re.compile(r'[^\s\w_]+').sub(' ', string)  # removes all non-alfanumeric characters
        string = string.lower()
        strings.append(string)
    return np.array(strings)


def main():
    """The main function of the program."""
    pointer = 1
    batch = 1

    print("Start storing all classes is seperate files.")
    stop = False
    while not stop:
        print("Now getting categories from batch", batch, "...")
        stop, data = load_data(pointer, BATCH_SIZE)
        classes = get_classes(data)
        strings = strings_from_data(data)
        labels = data[:, COLUMN_PRICE]
        store_classes(classes, strings, labels)

        pointer += BATCH_SIZE
        batch += 1


main()
