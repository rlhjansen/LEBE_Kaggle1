from __future__ import print_function
import os, csv, pickle

#
# verzameld alle merken uit de input file
# additional=False neemt eerdere input niet mee/overwrite oorspronkelijke file
# additional=True voegt nieuwe merken toe
#

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
# creëert een file waarin per item ofwel het merk ofwel de mogelijke merken genoteerd staan
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

infile = os.path.join(os.pardir, "trainColumnSwitched.tsv")
outfile = os.path.join(os.pardir, "possiblebrands.tsv")
dictloc = os.path.join(os.pardir, "newBrandDict")

"""
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

#
# Todo needs behaviour for shipping info
#

def gather_keywords(infile, outfile):
    wordset = set()
    with open(infile, encoding="utf8") as inf:
        infr = csv.reader(inf, delimiter="\t")
        with open(outfile, "w", encoding="utf8") as outf:
            writer = csv.writer(outf, delimiter='\t', skipinitialspace=True)
            for row in infr:
                row = [i.decode('utf-8').lower() for i in row]
                wordset.add(set(row[1].split(" ")))
                wordset.add(set(["condition"+row[2]]))
                wordset.add(set(row[3].split("/\\")))
                wordset.add(set(row[4].split(" ")))
                wordset.add(set(row[5].split(" ")))
            outf.close()
        inf.close()
    i = 0
    worddict = {}
    for word in wordset:
        worddict[word] = i
        i += 1
    with open(os.path.join(os.pardir, outfile), 'wb') as out:
        pickle.dump(worddict, out)
        out.close()


#
# Todo function needs to be written
#

def line_to_vec(line, dict)
    pass
