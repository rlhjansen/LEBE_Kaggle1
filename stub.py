import numpy as np
import re, csv
from collections import Counter

PATH_TRAIN = "../train.tsv"
PATH_TEST = "../test.tsv"
COLUMN_LABEL = 5
BATCH_SIZE = 1000  # Don't make this number much larger than 100000
DATA_SIZE = sum(1 for line in open(PATH_TRAIN))  # Number of \n in train file
THRESH = 50  # How often needs a word to occur before it is used
# This is a stub file, a file in which all (important) functions are explained.



COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_SHIPPING = 4
COLUMN_DESCRIPTION = 5


def sentence_to_words(data):
    """Gives each word in the array a unique int in a dictionary"""
    words = []
    for i in range(data.shape[0]):
        row = data[i]
        string = (row[COLUMN_NAME] + ' ' + row[COLUMN_BRAND] + ' ' +
                  row[COLUMN_DESCRIPTION])
        string = re.compile(r'[^a-zA-Z]+').sub(' ', string)

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


def splitdata(data, ratio=0.7, train_val=False):
    """Returns a training and a validation array"""
    rand_permutaion = np.random.permutation(len(data))
    rand_data = data[rand_permutaion]

    if train_val:
        i = int(data[:, -1].shape[0] * ratio)
        train_data = rand_data[:i, :-1]
        val_data = rand_data[i:, :-1]

        train_labels = rand_data[:i, -1]
        val_labels = rand_data[i:, -1]
        return train_data, train_labels, val_data, val_labels
    else:
        return rand_data[:, :-1], rand_data[:, -1]


def shipping_data(data):
    return np.array(data[:, COLUMN_SHIPPING])


def condition_data(data):
    return np.array(data[:, COLUMN_CONDITION])


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
def store_data(shipping_data, condition_data, words, labels, word_map, cats, cat_map, path):
    """Return a version of the data that a neural network can take as input."""
    w_keys = word_map.keys()
    c_keys = cat_map.keys()
    with open(path, "ab") as vec_file:
        for j in range(len(words)):

            in_row_w_string = "\t".join([str(word_map.get(word)) for word in words[j] if word in w_keys])
            in_row_cats = "\t".join([str(cat_map.get(cat)) for cat in cats[j] if cat in c_keys])
            in_row_shipping = "\t".join(shipping_data[j])
            in_row_condition = "\t".join(condition_data[j])
            in_row_label = labels[j]
            invec = ",".join([in_row_w_string, in_row_cats, in_row_shipping, in_row_condition, in_row_label])+"\n"
            print(invec)
            vec_file.write(invec)

        vec_file.close()
# in_vecs: [[word word word ... word shipping condition price] ... ]


def retrieve_data(readfile, pointer, batch_size, specificslist, kwargs):
    if kwargs.get("returntype") == "array":
        linevecs = [None]*batch_size
        for _ in range(pointer):
            readfile.readline()
        for j in range(batch_size):
            rawline = readfile.readline()
            print(len(rawline))

            comma_processed = rawline.split(",")
            k = 0
            for inst in comma_processed:
                print(inst)
                k += 1

            alpha_processed = [comma_processed[0].split("\t"),
                               comma_processed[1].split("\t")]
            numeric_processed = [int(comma_processed[2]),
                                 int(comma_processed[3]),
                                 int(comma_processed[4])]
            linelist = alpha_processed.extend(numeric_processed)
            linevecs[j] = translate_line_input(linelist, specificslist, kwargs)
            print(linelist)
        return np.array(linevecs)


def save_specifics(num_of_words, num_of_cats, filepath):
    with open(filepath, "wb") as specfile:
        specfile.write(str(num_of_words) + "\n")
        specfile.write(str(num_of_cats) + "\n")
        specfile.close()


def get_specifics_from_file(filepath):
    specifics = []
    with open(filepath, "rb") as specfile:
        line = specfile.readline()
        specifics.append(int(line[:-1]))
        line = specfile.readline()
        specifics.append(int(line[:-1]))
    return specifics


def get_line_parameters(specifics, _dict):
    if _dict.get("all"):
        vec_size = specifics[0]+specifics[1]+1+5+1
        cat_start = specifics[0]
        ship_loc = specifics[0]+specifics[1]
        condition_start = specifics[0]+specifics[1]+1
        price_loc = specifics[0]+specifics[1]+1+5
        return [cat_start, ship_loc, condition_start, price_loc, vec_size]
    # Todo add supplementary for partial execution
    length = 0
    returnlist = [None]*5
    if _dict.get("words"):
        length += specifics[0]
    if _dict.get("categories"):
        returnlist[0] = length
        length += specifics[1]
    if _dict.get("shipping"):
        returnlist[1] = length
        length += 1
    if _dict.get("condition"):
        returnlist[2] = length
        length += 5


    if _dict.get("price"):
        returnlist[-2] = length
        length +=1
    # vector length
    returnlist[-1] = length
    return returnlist


# specificslist: cat_start, ship_loc, condition_start, price_loc, vec_size
# line_list: [[wordindexes], [catindexes], shipping, condition, price]
# line adds everything to the input vector, otherwise one can use differen keywords
# setting the following keywords to true will result in imput vectors of those parts.
# keywords: "words", "categories", "shipping"
def translate_line_input(line_list, specificslist, _dict):

    npvec = np.zeros((specificslist[-1]), dtype=float)
    """ for working with all of the data """
    if _dict.get("all"):
        npvec = np.zeros((specificslist[-1]), dtype=float)
        word_array = np.array(line_list[0])
        cat_array = np.array(line_list[1])+specificslist[0]
        condition_array = np.array(specificslist[2]+line_list[3])
        if line_list[2]:
            ship_loc = specificslist[1]
            npvec[ship_loc] = 1.0
        npvec[word_array] = 1.0
        npvec[cat_array] = 1.0
        npvec[condition_array] = 1.0
        npvec[specificslist[-2]] = line_list[-1]

    """ for working with parts of the data"""
    if _dict.get("words", False):
        npvec[np.array(line_list[0])] =1.0
    if _dict.get("categories", False):
        npvec[np.array(line_list[1])+specificslist[0]] = 1.0
    if _dict.get("shipping", False):
        if line_list[2]:
            ship_loc = specificslist[1]
            npvec[ship_loc] = 1.0
    if _dict.get("condition", False):
        npvec[np.array(specificslist[2]+line_list[3])] = 1.0
    if _dict.get("price", False):
        npvec[specificslist[-2]] = line_list[-1]

    return npvec


#Todo add categorical to above

# start: int; How many lines to skip before reading the data file.
# size: int; How many lines to read.
def load_data(start, size):
    """Return the trainingdata as an array."""
    data = []
    labels = []
    continues = True
    if start + size > DATA_SIZE:
        size = DATA_SIZE - start
        continues = False
    with open(PATH_TRAIN, "rb") as f:
        for _ in range(start):
            f.readline()
        for _ in range(size):
            line = f.readline()
            line = line.split("\t")
            label = line[COLUMN_LABEL]
            line = line[1:COLUMN_LABEL] + line[COLUMN_LABEL+1:]
            data.append(np.array(line))
            labels.append(label)

    return np.array(data), np.array(labels), continues
# data: n x 6 array; 
# label: n x 1 array; prices


# returns a dictionary with vector indexes for each word that appears
# more than the threshold value
def cut_counter(counter, thresh=0):
    new_dict = {}
    i = 0
    for key in counter.keys():
        if counter.get(key) > thresh:
            new_dict[key] = i
            i += 1
    return new_dict


# start: int; How many lines to skip before reading the data file.
# size: int; How many lines to read.
def load_input(start, size):
    """Return input vectors as an 2-dim array."""
    pass
# data
# label: n x 1 array; prices


# prepares variables for a new run through the Kaggle data.
def prep():
    batch_count = 0
    pointer = 0
    word_counter = Counter()
    cat_counter = Counter()
    print("Batch size =", BATCH_SIZE, "Data size =", DATA_SIZE)

    data, labels, continues = load_data(pointer, BATCH_SIZE)
    data = data[1:]
    labels = labels[1:]
    PATH_INPUT_VECTOR = derive_filename_storage()
    return batch_count, pointer, word_counter, cat_counter, data, labels, continues, PATH_INPUT_VECTOR


# This is the main function of the preprocessing.py
def main(**kwargs):
    """The main function of the program"""
    if kwargs.get("restart"):
        batch_count, pointer, word_counter, cat_counter, data, _, continues, PATH_INPUT_VECTOR = prep()
        for i in kwargs.get("extra_features"):
            pass
        while continues:
            """ Dictionaries for words & categories created """
            words = sentence_to_words(data)  # process name, brand, description
            word_counter = count_words(words,
                                       word_counter)  # return a counter of all words
            cats = row_to_cats(data)
            cat_counter = count_cats(cats,
                                     cat_counter)  # return a counter of all categories

            print("Now indexed batch", batch_count, "...")
            pointer += BATCH_SIZE
            data, _, continues = load_data(pointer, BATCH_SIZE)
            batch_count += 1
            break

        """ Create dictionaries for common words & categories """
        word_map = cut_counter(word_counter, thresh=THRESH)
        cat_map = cut_counter(cat_counter)

        batch_count, pointer, _, _, data, labels, continues, PATH_INPUT_VECTOR = prep()
        path_store = derive_filename_storage()
        path_specifics = path_store + "specifics"
        save_specifics(len(word_map.keys()), len(cat_map.keys()),
                       path_specifics)
        specifics = get_specifics_from_file(path_specifics)
        specificslist = get_line_parameters(specifics, kwargs)
        while continues:
            """ saving rows in new format """
            words = sentence_to_words(data)  # process name, brand, description
            cats = row_to_cats(data)
            ships_data = shipping_data(data)
            cond_data = condition_data(data)
            store_data(ships_data, cond_data, words, labels, word_map, cats,
                       cat_map, path_store)
            # print(in_vecs)
            data, labels, continues = load_data(pointer, BATCH_SIZE)
            print("done with batch", batch_count)
            batch_count += 1

            # print("Now starting batch", batch_count, "...")
            pointer += BATCH_SIZE

    path_store = derive_filename_storage()
    path_specifics = path_store + "specifics"
    specifics = get_specifics_from_file(path_specifics)
    specificslist = get_line_parameters(specifics, kwargs)

    """ Returns data line by line """

    batch_count, pointer, _, _, data, labels, continues, PATH_INPUT_VECTOR = prep()
    while continues:
        with open(path_store, 'rb') as readfile:
            print("opening from", path_store)
            np_vec = retrieve_data(readfile, pointer, BATCH_SIZE, specificslist, kwargs)
            print(np_vec)
            #train_data, train_labels, val_data, val_labels = splitdata(data, ratio=0.7, train_val=True)
            X, Y = splitdata(data)
            if kwargs.get("test"):
                break


# returns string with path for specific run values
# retval = "../invec_tresh50.tsv
def derive_filename_storage():
    base = "../invec_"
    threshstring = "tresh"+str(THRESH)
    end = ".tsv"
    retval = base+threshstring+end
    return retval



# arr: any 2-dim array
# path: string; path to storage file
# pointer: int; from which byte in the file do you start storing the "arr"


# pointer: int; at which byte did you stop storing
# how to store: word word word ... word shipping condition price\n ... (use tabs instead of spaced)


# restart=True when using different threshold for the first time only
# test for testing first <batch_size> input vectors
# returntype="array" if you want data in form of a numpy array of <batch_size>
# all=True if you want to gain input data for all input types
# (otherwise check retrieve_data() for other keyword args)
main(extra_features=[], all=True, restart=True, test=True, returntype="array")

# if other functionalities are added, like NLTK, look up derive_filename_storage()
#
