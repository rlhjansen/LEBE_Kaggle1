import os, csv


#TODO: write application to add in addition to previosu found brands
def write_brands_to_file(file, outname, additional=False):
    brands = set()
    if additional == True:
        pass
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



write_brands_to_file("train.tsv", "trainbrands.txt")

