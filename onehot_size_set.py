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
    """example input: INFILE, OUTFILE, origin='standard', cutoff=4, returntype="dict"""
    worddict = {}
    with open(INFILE) as inf:
        infr = csv.reader(inf, delimiter="\t")
        for row in infr:
            row = [i.lower() for i in row]
            row = convert_alphanumerical(row)
            res = extract_line_words_count(worddict, row, origin=kwargs.get("origin"), returntype="dict")
            worddict = res
        inf.close()
    i = 0
    worddictnums = {}
    for key in worddict.keys():
        if worddict[key] > kwargs.get("cutoff"):
            worddictnums[key] = i
            #print(key, i)
            i += 1
            #print("kwargs.get('cutoff')", kwargs.get("cutoff"))
    with open(OUTFILE, 'wb') as out:
        for key in worddictnums.keys():
            #print(key, worddictnums.get(key))
            pass
        pickle.dump(worddictnums, out)
        print("saves")


def add_list_to_dict(_dict, list):
    """Add all words in a list to a dictionary."""
    for word in list:
        if not _dict.get(word):
            _dict[word] = 1
        else:
            _dict[word] += 1
    return _dict

#
# if kwargs.get("returntype") == "dict")
#   returns dictionary for line with k:v = word:#
# else:
#   returns set for line with used words.
def extract_line_words_count(worddict, row, **kwargs):
    """
    # if kwargs.get("returntype") == "dict")
    #   returns dictionary for line with k:v = word:#
    # else:
    #   returns set for line with used words."""
    if kwargs.get("returntype") == "dict":
        _list = []
        for inst in [row[1].split(" "), ["condition" + row[2]], re.split("/|", row[3]), row[4].split(" "), [row[4]], row[5].split(" "), ["shippingYes" if row[6] == 1 else "shippingNo"]]:
            _list.extend(inst)
        mydict = add_list_to_dict(worddict, _list)
        return mydict
    else:
        return extranct_line_features(row)


def extranct_line_features(row):
    wordset = set()
    wordset |= set(row[1].split(" "))  # woorden uit de naam als input
    wordset |= set(
        ["condition" + row[2]])  # conditions als input (1,2,3,4,5)
    wordset |= set(re.split("/|", row[3]))  # categorien als input
    wordset |= set(row[4].split(
        " "))  # woorden uit merknamen als input (voor wanneer merk uit beschrjving gehaald zou kunnen worden)
    wordset |= set([row[4]])  # gehele merken als input
    wordset |= set(row[5].split(" "))  # woorden uit de beschrijving
    wordset |= set(["shippingYes" if row[6] == 1 else "shippingNo"])
    return wordset

def features_to_input(wordset, _dict, size):
    whole = np.array([0.0]*size)
    try:
        ones = [_dict.get(word, 0) for word in wordset]
    except:
        raise("you're not using the right dictionary")
    whole[ones] = 1.0
    return whole



def convert_to_npdata(infile_iterator, _dictionary, batchsize=1000, **kwargs):
    data = []
    labels = []
    continuation = True
    for _ in range(batchsize):
        try:
            row = infile_iterator.next()
        except StopIteration:
            continuation = False
            break
        row = [i.lower() for i in row]
        row = convert_alphanumerical(row)
        wordset = extract_line_words_count({}, row, returntype=kwargs.get("returntype"))
        try:
            data.append(wordset)
            labels.append(row[7])
        except(MemoryError):
            print("Use a smaller batch size")
            break

    labels_array = np.array(labels)
    return np.array(data), labels_array, continuation




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

def train_with_batches(inputfile, batch_size, restart=True, **kwargs):
    idstring = "feature_cut_"+str(kwargs.get("cutoff"))+" "+kwargs.get("returntype")
    dictionaryfile = os.path.join(os.pardir, idstring)
    if restart:
        gather_keywords_cutoff(inputfile, dictionaryfile, origin=kwargs.get("origin"), cutoff=kwargs.get("cutoff", 0))
    with open(inputfile) as inf:
        with open(dictionaryfile) as kd:
            continuation = True
            infr = csv.reader(inf, delimiter="\t")
            lkd = pickle.load(kd)
            while continuation:
                data, labels, continuation = convert_to_npdata(infr, lkd, batchsize=batch_size, returntype=kwargs.get("returntype"))
                if kwargs.get("returntype") == "dict":
                    pass
                else:
                    for s in data:
                        pass
                train_data, train_labels, val_data, val_labels = splitdata(data, labels)
                print("runs completely")
                #
                # some machine learning function
                #
        numpy_array = np.array(data)
        print("finito")

inputfile = os.path.join(os.pardir, "trainColumnSwitched.tsv")
train_with_batches(inputfile, 500, restart=True, origin="standard", cutoff=1, returntype="dict")