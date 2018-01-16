from __future__ import print_function
import os, csv, pickle, re
import numpy as np


# regex to clean data
NAME_DISCR_PATTERN = re.compile(r'[^\s\w_]+')  # What will remain
NUMBER_PATTERN = re.compile(r'[^\s\w_.]+')  # What will remain
CATEGORY_PATTERN = re.compile(r'[^\s\w_./\'&,]+')  # What will remain


def convert_alphanumerical(row):
    """use regeox on row"""
    return [NUMBER_PATTERN.sub('', row[0]),
                   NAME_DISCR_PATTERN.sub(' ', row[1]),
                   NUMBER_PATTERN.sub('', row[2]),
                   CATEGORY_PATTERN.sub('', row[3]),
                   NAME_DISCR_PATTERN.sub(' ', row[4]),
                   NAME_DISCR_PATTERN.sub(' ', row[5]), row[6],
                   NUMBER_PATTERN.sub('', row[7])]


def gather_keywords_cutoff(INFILE, OUTFILE, **kwargs):
    """example input: INFILE, OUTFILE, origin='standard', cutoff=4"""
    worddict = {}
    with open(INFILE) as inf:
        infr = csv.reader(inf, delimiter="\t")
        for row in infr:
            row = [i.lower() for i in row]
            row = convert_alphanumerical(row)
            extract_line_words_count(worddict, row, kwargs.get("origin"))
        inf.close()
    i = 0
    worddictnums = {}
    for key in worddict.keys():
        if worddict[key] >= kwargs.get("cutoff"):
            worddictnums[key] = i
            i += 1
    kwargstring = "".join([str(kwargs[key]) for key in kwargs.keys()])
    OUTFILE += kwargstring
    with open(OUTFILE, 'wb') as out:
        pickle.dump(worddictnums, out)


def add_list_to_dict(_dict, list):
    """Add all words in a list to a dictionary."""
    for word in list:
        if not _dict.get(word):
            _dict[word] = 1
        else:
            _dict[word] += 1


def extract_line_words_count(worddict, row, _list):
    for key in _list:
        if key == "standard":
            add_list_to_dict(worddict, row[1].split(" "))  # woorden uit de naam als input
            add_list_to_dict(worddict, ["condition" + row[2]])  # conditions als input (1,2,3,4,5)
            add_list_to_dict(worddict, re.split("/|", row[3]))  # categorien als input
            add_list_to_dict(worddict, row[4].split(" "))  # woorden uit merknamen als input (voor wanneer merk uit beschrjving gehaald zou kunnen worden)
            add_list_to_dict(worddict, [row[4]])  # gehele merken als input
            add_list_to_dict(worddict, row[5].split(" "))  # woorden uit de beschrijving
            add_list_to_dict(worddict,
                             ["shippingYes" if row[6] == 1 else "shippingNo"])
            break
        if key == "name":
            add_list_to_dict(worddict, row[1].split(" "))  # woorden uit de naam als input
        if key == "condition":
            add_list_to_dict(worddict, ["condition" + row[2]])  # conditions als input (1,2,3,4,5)
        if key == "category":
            add_list_to_dict(worddict, re.split("/|", row[3]))  # categorien als input
        if key == "brandWords":
            add_list_to_dict(worddict, row[4].split(
                " "))  # woorden uit merknamen als input (voor wanneer merk uit beschrjving gehaald zou kunnen worden)
        if key == "brand":
            add_list_to_dict(worddict, row[4])  # gehele merken als input
        if key == "descrWords":
            add_list_to_dict(worddict, row[5].split(" "))  # woorden uit de beschrijving
        if key == "shipping":
            add_list_to_dict(worddict,
                             ["shippingYes" if row[6] == 1 else "shippingNo"])
    return worddict


def extranct_line_features(row, *args):
    wordset = set()
    for key in args:
        if key == "standard":
            wordset |= set(row[1].split(" "))  # woorden uit de naam als input
            wordset |= set(["condition" + row[2]])  # conditions als input (1,2,3,4,5)
            wordset |= set(re.split("/|", row[3]))  # categorien als input
            wordset |= set(row[4].split(" "))  # woorden uit merknamen als input (voor wanneer merk uit beschrjving gehaald zou kunnen worden)
            wordset |= set([row[4]])  # gehele merken als input
            wordset |= set(row[5].split(" "))  # woorden uit de beschrijving
            wordset |= set(["shippingYes" if row[6] == 1 else "shippingNo"])
            break
        if key == "name":
            wordset |= set(row[1].split(" "))  # woorden uit de naam als input
        if key == "condition":
            wordset |= set(["condition" + row[2]])  # conditions als input (1,2,3,4,5)
        if key == "category":
            wordset |= set(re.split("/|", row[3]))  # categorien als input
        if key == "brandWords":
            wordset |= set(row[4].split(
                " "))  # woorden uit merknamen als input (voor wanneer merk uit beschrjving gehaald zou kunnen worden)
        if key == "brand":
            wordset |= set(row[4])  # gehele merken als input
        if key == "descrWords":
            wordset |= set(row[5].split(" "))  # woorden uit de beschrijving
        if key == "shipping":
            wordset |= set(["shippingYes" if row[6] == 1 else "shippingNo"])
    return wordset


def features_to_input(wordset, _dict, size):
    """Change a wordset into an array compatible with the algorithm."""
    whole = np.array([0.0]*size)
    try:
        ones = np.array([_dict.get(word, 0) for word in wordset])
    except:
        raise("you're not using the right dictionary")
    whole[ones] = 1.0
    return whole



def convert_to_npdata(infile_iterator, _dictionary, batchsize=1000):
    data = []
    labels = []
    size = len(_dictionary)
    continuation = True
    for _ in range(batchsize):
        try:
            row = infile_iterator.next()
        except StopIteration:
            continuation = False
            break
        row = [i.decode('utf-8').lower() for i in row]
        row = convert_alphanumerical(row)
        wordset = extranct_line_features(row, "standard")
        try:
            data.append(features_to_input(wordset, _dictionary, size))
            labels.append(row[7])
        except(MemoryError):
            print("Use a smaller batch size")
            print(len(data))
            break
        print(features_to_input(wordset, _dictionary, size))

    data_array = np.array(data)
    labels_array = np.array(labels)
    print(np.sum(data_array, axis=0))
    return data_array, labels_array, continuation



INFILE = os.path.join(os.pardir, "trainColumnSwitched.tsv")
OUTFILE = os.path.join(os.pardir, "keyword_dict")

gather_keywords_cutoff(INFILE, OUTFILE, origin="standard", cutoff=4)
print(convert_to_npdata(INFILE, OUTFILE))


def splitdata(data, labels, ratio=0.7):
    """Returns a training and a validation array"""
    i = int(labels.shape[0] * ratio)
    rand_permutaion = np.random.permutation(len(data))
    rand_data = data[rand_permutaion]
    train_data = rand_data[:i]
    val_data = rand_data[i:]

    rand_labels = labels[rand_permutaion]
    train_labels = rand_labels[:i]
    val_labels = rand_labels[i:]
    return train_data, train_labels, val_data, val_labels


def train_with_batches(inputfile, dictionaryfile, batch_size):
    with open(inputfile) as inf:
        with open(dictionaryfile) as kd:
            continuation = True
            infr = csv.reader(inf, delimiter="\t")
            lkd = pickle.load(kd)
            size = len(lkd)
            j = 0
            while continuation:
                data, labels, continuation = convert_to_npdata(infr, lkd, batchsize=batch_size)
                train_data, train_labels, val_data, val_labels = splitdata(data, labels)
                #
                # some machine learning function
                #
        numpy_array = np.array(data)
        print(np.sum(numpy_array, axis=0))

