from __future__ import print_function
import os, csv, pickle, re
import numpy as np

#
# verzameld alle inputvariabelen uit de input file
#

#
# regex to clean data
#
name_discr_pattern = re.compile(r'[^\s\w_]+') # What will remain
number_pattern = re.compile(r'[^\s\w_.]+') # What will remain
category_pattern = re.compile(r'[^\s\w_./\'&,]+') # What will remain


# uses regex on rows to only leave relevant characters per characteristic (letters for words etc.)
def convert_alphanumerical(row):
    return [number_pattern.sub('', row[0]),
                   name_discr_pattern.sub(' ', row[1]),
                   number_pattern.sub('', row[2]),
                   category_pattern.sub('', row[3]),
                   name_discr_pattern.sub(' ', row[4]),
                   name_discr_pattern.sub(' ', row[5]), row[6],
                   number_pattern.sub('', row[7])]


# uses decode to be able to work with utf8
# converts all charters to lowercase
# uses all words in the training data
def gather_keywords(infile, outfile, *args):
    wordset = set()
    with open(infile) as inf:
        infr = csv.reader(inf, delimiter="\t")
        for row in infr:
            row = [i.decode('utf-8').lower() for i in row]
            row = convert_alphanumerical(row)
            wordset |= (extranct_line_features(row, *args))
        inf.close()
    i = 0
    worddict = {}
    for word in wordset:
        worddict[word] = i
        i += 1
    for key in worddict:
        print(key)
    with open(outfile, 'wb') as out:
        pickle.dump(worddict, out)



# returns a set of all words relevant for that line
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
    whole = np.array([0.0]*size)
    try:
        ones = np.array([_dict[word] for word in wordset])
    except:
        raise("you're not using the right dictionary")
        exit(1)
    whole[ones] = 1.0
    return whole


#
# create dictionary
#

infile = os.path.join(os.pardir,"trainColumnSwitched.tsv")
outfile = os.path.join(os.pardir, "keyword_dict")
gather_keywords(infile, outfile, "standard")


infile = os.path.join(os.pardir,"trainColumnSwitched.tsv")
outfile = os.path.join(os.pardir, "keyword_dict")
BATCHSIZE = 10

#
# converts to a numpy data array, still working on, batchsize is incredibly small
#
def convert_to_npdata(infile, outfile):
    data = []
    with open(infile) as inf:
        print("test1")
        infr = csv.reader(inf, delimiter="\t")
        with open(outfile, "rb") as kd:
            print("test2")
            lkd = pickle.load(kd)
            size = len(lkd)
            j = 0
            for key in lkd:
                print("test3")
                pass
            for row in infr:
                print("test4")
                row = [i.decode('utf-8').lower() for i in row]
                row = convert_alphanumerical(row)
                wordset = (extranct_line_features(row, "standard"))
                try:
                    data.append(features_to_input(wordset, lkd, size))
                except(MemoryError):
                    print("Use a smaller batch size")
                    print(len(data))
                    break
                #print(features_to_input(wordset, lkd, size))
                j += 1
                if j == BATCHSIZE:
                    print("test5")
                    break
    return np.array(data)

print(convert_to_npdata(infile, outfile))



