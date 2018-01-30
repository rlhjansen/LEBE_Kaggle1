"""This program will calculate which words are used very different between
classes."""
from __future__ import print_function, division
from pathlib import Path
import re
from collections import Counter
import cPickle
import numpy as np
from nltk.corpus import stopwords

# Important parameters on how the program is run.
BATCH_SIZE = 100000  # Don't make this much larger than 100000
THRESH = 100
VEC_LEN = 100
TRAIN_RATIO = 0.8  # The ratio train / total
CLASS_DEPTH = 2  # We don't take very niche subcategories.

# The input files
PATH_TRAIN = "../train.tsv"
PATH_CLASSES = "../split_classes/"

# The output file
PATH_OUT = "../delta_vec_" + str(VEC_LEN) + ".tsv"
PATH_LMAP = "../delta_labels_" + str(VEC_LEN) + ".tsv"
PATH_DMAP = "../delta_occ_" + str(VEC_LEN) + ".tsv"
PATH_OUT_TRAIN = "../delta_train_" + str(VEC_LEN) + ".tsv"
PATH_OUT_VAL = "../delta_val_" + str(VEC_LEN) + ".tsv"

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


def delta_function(word_occ, tot_word_occ, word):
    """The function determining which words are least evenly distributed among
    classes."""
    return abs(word_occ[word] - tot_word_occ[word])


def dwords_from_data(data, class_names, delta_vec, delta_map):
    """Return an array of vectors with the delta word-occurence for each row."""
    delta_words = []
    for i, row in enumerate(data):
        string = (row[COLUMN_NAME] + ' ' + row[COLUMN_BRAND] + ' ' +
                  row[COLUMN_DESCRIPTION])
        string = re.compile(r'[^\s\w_]+').sub(' ', string)  # removes all non-alfanumeric characters
        string = string.lower()
        word_arr = np.array(string.split(' '))
        dword_vec = - delta_map[str(class_names[i])]
        for word in word_arr:
            if word in delta_vec:
                dword_vec[delta_vec == word] += 1.0
        delta_words.append(dword_vec)
    return np.array(delta_words)


def get_class(data):
    """Return an array of class names from the data."""
    conditions = data[:, COLUMN_CONDITION]
    categories = data[:, COLUMN_CATEGORY]
    shipping = data[:, COLUMN_SHIPPING]

    classes = []
    for i, _ in enumerate(conditions):
        name = ""
        name += conditions[i] + '_'

        category = categories[i].split('/')
        category = category[:CLASS_DEPTH]
        for cat in category:
            name += cat + '-'
        name = name[:-1] + '_'

        name += shipping[i]
        classes.append(name)
    return np.array(classes)


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


def labels_from_data(data, class_names, label_map):
    """Return an array with the prizes matching the rows"""
    labels = []
    for i, row in enumerate(data):
        label = np.array([0.0, 0.0])
        label[0] = label_map[class_names[i]]
        label[1] = row[COLUMN_PRICE]
        labels.append(label)
    return labels


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


def load_delta(path):
    """Return the delta vector stored at 'path'"""
    delta_vec = []
    with open(path) as f:
        for word in f:
            delta_vec.append(word[:-1])
    return np.array(delta_vec)


def load_pickle(path):
    """Return the python object located at 'path' using cPickle"""
    obj = None
    with open(path) as f:
        obj = cPickle.load(f)
    return obj


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


def make_delta():
    """Create a vector with words scoring highest in the delta function."""
    tot_word_occ = count_data(PATH_TRAIN)
    delta_occ = init_delta_occ(tot_word_occ)

    pathlist = Path(PATH_CLASSES)
    print("Start comparing words with the classes.")

    for path_obj in pathlist.iterdir():
        class_name = get_class_name(str(path_obj))
        print("Comparing words from \"" + class_name + "\"")
        word_occ, class_labels = count_words(path_obj)
        for word in word_occ:
            delta_occ[word] += delta_function(word_occ, tot_word_occ, word)

    print("\nBiggest delta words:")
    max_deltas = sorted(delta_occ, key=delta_occ.__getitem__, reverse=True)
    max_deltas = max_deltas[:VEC_LEN]
    for word in max_deltas:
        print(word, delta_occ[word])
    return np.array(max_deltas)


def map_delta_occ(delta_vec):
    """Return a dictionary that matches class names with an array with the
    relative occurance of each word in the delta vec. Also make """
    delta_map = dict()
    label_map = dict()

    print("Start mapping the delta occurance with each class.")
    pathlist = Path(PATH_CLASSES)
    for path_obj in pathlist.iterdir():
        class_name = get_class_name(str(path_obj))
        delta_occ = np.zeros(len(delta_vec))
        num_rows = 0
        mean_label = 0

        print("Now mapping:", '"' + class_name + '"')
        pointer = 0
        stop = False
        while not stop:
            stop, words, labels = load_words(path_obj, pointer, BATCH_SIZE)
            for row in words:
                num_rows += 1
                for word in row:
                    delta_occ[delta_vec == word] += 1

            mean_label += sum(labels)
            pointer += BATCH_SIZE
        
        mean_label = mean_label / num_rows
        label_map[class_name] = mean_label
        delta_occ = delta_occ / num_rows
        delta_map[class_name] = delta_occ
    
    return delta_map, label_map


def normalize_counter(num_words, word_count):
    """Return a dictionary that maps a word with its relative use."""
    word_occ = dict()
    for word in word_count.keys():
        word_occ[word] = word_count[word] / num_words
    return word_occ


def preprocess(path_in, path_train, path_val, delta_vec, delta_map, label_map):
    """Make and store vectors that a neural network can use."""
    batch = 1
    pointer = 1 + BATCH_SIZE * 2
    stop = False
    while not stop:
        print("\nNow loading batch", batch, "...")
        stop, data = load_data(path_in, pointer, BATCH_SIZE)
        train, val = split_train_val(data, TRAIN_RATIO)

        print("... Extracting class values ...")
        class_train = get_class(train)
        class_val = get_class(val)
        dwords_train = dwords_from_data(train, class_train, delta_vec, delta_map)
        dwords_val = dwords_from_data(val, class_val,  delta_vec, delta_map)
        labels_train = labels_from_data(train, class_train, label_map)
        labels_val = labels_from_data(val, class_val, label_map)

        print("... Converting to input ...")
        in_train = to_input(dwords_train, labels_train)
        in_val = to_input(dwords_val, labels_val)

        print("... Storing")
        store_arr(path_train, in_train)
        store_arr(path_val, in_val)

        batch += 1
        pointer += BATCH_SIZE


def split_train_val(data, ratio):
    """Return a set for training and a set for validation"""
    np.random.shuffle(data)
    index = int(len(data) * ratio)
    train = data[:index]
    val = data[index:]
    return train, val


def store_arr(path, arr):
    """Store an array in a file given by path"""
    with open(path, 'a') as f:
        for row in arr:
            f.write((row) + '\n')


def store_pickle(path, obj):
    """Store an object at the location given with 'path' using cPickle"""
    with open(path, 'wb') as f:
        cPickle.dump(obj, f)


def store_vec(vec):
    """Store the delta vector in a file."""
    with open(PATH_OUT, 'w') as f:
        for word in vec:
            f.write(word + '\n')


def to_input(words, labels):
    """Return an array with strings that will be stored in the input file."""
    in_vec = []
    for i, row in enumerate(words):
        in_str = ""
        for elem in row:
            in_str += str(elem) + '\t'
        for label in labels[i]:
            in_str += str(label) + '\t'
        in_str = in_str[:-1]
        in_vec.append(in_str)
    return np.array(in_vec)


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


def main(vec_path=False, dmap_path=False, lmap_path=False):
    """The main function of the program"""
    delta_vec = np.array([])
    if vec_path:
        print("Loading the delta from path", '"' + vec_path + '"')
        delta_vec = load_delta(vec_path)
    else:
        print("Creating a new delta")
        delta_vec = make_delta()
        store_vec(delta_vec)

    delta_map = dict()
    label_map = dict()
    if vec_path and dmap_path and lmap_path:
        print("\nLoading the delta map and the label map")
        delta_map = load_pickle(dmap_path)
        label_map = load_pickle(lmap_path)
    else:
        print("\nCreating a delta map and a label map")
        delta_map, label_map = map_delta_occ(delta_vec)
        store_pickle(PATH_DMAP, delta_map)
        store_pickle(PATH_LMAP, label_map)

    print("\nPreprocessing the data")
    preprocess(PATH_TRAIN, PATH_OUT_TRAIN, PATH_OUT_VAL, delta_vec, delta_map, label_map)


main()
