from __future__ import print_function
import os, csv, pickle, re
import numpy as np

#
# verzameld alle merken uit de input file
# additional=False neemt eerdere input niet mee/overwrite oorspronkelijke file
# additional=True voegt nieuwe merken toe
#


name_discr_pattern = re.compile(r'[^\s\w_]+') # What will remain
number_pattern = re.compile(r'[^\s\w_.]+') # What will remain
category_pattern = re.compile(r'[^\s\w_./\'&,]+') # What will remain


def write_brands_to_file(file, outname, additional=False):
    brands = set()
    if additional == True:
        with open(os.path.join(os.pardir, outname), 'r') as reading:
            for line in reading:
                print(line[:-1])
                brands.add(line[:-1])
            reading.close()
    with open(os.path.join(os.pardir, file), encoding="utf8") as tsvin:
        tsvs = csv.reader(tsvin, delimiter="\t")
        for row in tsvs:
            brands.add(row[4])
        tsvin.close()
    brandl = list(brands)
    brandl.sort()
    with open(os.path.join(os.pardir, outname), "w", encoding="utf8") as out:
        for brand in brandl:
            out.write(brand + "\n")
        out.close()


#write_brands_to_file("train.tsv", "trainbrands.txt", additional=True)


#
# voegt woordencombinatie van een merk (["air", "jordan])
# aan de dictionary van merken toe
#

def add_combination(_dict, brand):
    brand_parts = brand.split(" ")
    first = brand_parts[0]
    if not _dict.get(first):
        _dict[first] = [brand_parts]
    else:
        further = _dict.get(first)
        further.append(brand_parts)
        _dict[first] = further
    return _dict


#
# maakt een dictionary die alle eerste woorden van merken mapt naar het hele merk
# gesplit op basis van spatie (voor merkherkenning in beschrijving)
#

def make_brand_tree_dict(infile, outfile):
    brandpart_dict = {}
    with open(os.path.join(os.pardir, infile), 'r') as reading:
        for line in reading:
            #print(line[:-1])
            brandpart_dict = add_combination(brandpart_dict, line[:-1])
        reading.close()
    with open(os.path.join(os.pardir, outfile), 'wb') as out:
        pickle.dump(brandpart_dict, out)
        out.close()

#make_brand_tree_dict("trainbrands.txt", "newBrandDict")


#
# screent een string op mogelijke merken en returned deze
#

def recognize_brands(_partdict, descrline):
    lwords = descrline.split(" ")
    possibles = []
    for i in range(len(lwords)):
        progressions = _partdict.get(lwords[i])
        #print(lwords[i], "progressions", progressions)
        usable = True
        if progressions == None:
            continue
        for instance in progressions:
            for j in range(len(instance)):
                if not lwords[i+j] == instance[j]:
                    usable = False
                    break
            if usable:
                possibles.append(instance)
            usable = True
    return possibles


#
# creert een file waarin per item ofwel het merk ofwel de mogelijke merken genoteerd staan
# geschrapt
#
def write_possible_brands(infile, outfile, _dict):
    with open(infile, encoding="utf8") as inf:
        infr = csv.reader(inf, delimiter="\t")
        with open(outfile, "w", encoding="utf8") as outf:
            writer = csv.writer(outf, delimiter='\t', skipinitialspace=True)
            for row in infr:
                try:
                    if row[4] == "":
                        print(row)
                        possible_brands = recognize_brands(_dict, row[5])
                        print(possible_brands)
                        writer.writerow(str(possible_brands))
                    else:
                        writer.writerow(row[4])
                except IndexError:
                    continue
            outf.close()
        inf.close()



"""
infile = os.path.join(os.pardir, "trainColumnSwitched.tsv")
outfile = os.path.join(os.pardir, "possiblebrands.tsv")
dictloc = os.path.join(os.pardir, "newBrandDict")

with open(dictloc, "rb") as mydict:
    mydictLoaded = pickle.load(mydict)
    write_possible_brands(infile, outfile, mydictLoaded)
    mydict.close()

"""


#
# voorbeeld waarom merkherkenning tot problemen leid
#

"""
string = "This keyboard is in great condition and works like it came out of the box. All of the ports are tested and work perfectly. The lights are customizable via the Razer Synapse app on your PC."

dictloc = os.path.join(os.pardir, "newBrandDict")

with open(dictloc, "rb") as mydict:
    mydictLoaded = pickle.load(mydict)
    for key in mydictLoaded:
        print(key, mydictLoaded.get(key))
    print(recognize_brands(mydictLoaded, string))
    mydict.close()


#"""

def convert_alphanumerical(row):
    return [number_pattern.sub('', row[0]),
                   name_discr_pattern.sub(' ', row[1]),
                   number_pattern.sub('', row[2]),
                   category_pattern.sub('', row[3]),
                   name_discr_pattern.sub(' ', row[4]),
                   name_discr_pattern.sub(' ', row[5]), row[6],
                   number_pattern.sub('', row[7])]

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


with open(infile) as inf:
    infr = csv.reader(inf, delimiter="\t")
    with open(outfile, "rb") as kd:
        lkd = pickle.load(kd)
        size = len(lkd)
        j = 0
        for key in lkd:
            print(key)
        for row in infr:
            row = [i.decode('utf-8').lower() for i in row]
            row = convert_alphanumerical(row)
            wordset = (extranct_line_features(row, "standard"))
            print(features_to_input(wordset, lkd, size))
            j += 1
            if j == 100:
                break
