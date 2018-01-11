import csv
import os


# example:
# change_to_csv("test.tsv", "test.csv")
# result = csv file van deze data
# was nodig voor Knime, heeft voorlopig geen nut meer.
#

def change_to_csv(filename, outname, enc="utf8"):
    with open(os.path.join(os.pardir, filename), encoding=enc) as tsvin:
        with open(os.path.join(os.pardir, outname), 'w', encoding=enc) as csvout:
            tsvs = csv.reader(tsvin, delimiter='\t')
            for row in tsvs:
                newrow = ",".join(row) + "\n"
                csvout.write(newrow)
            csvout.close()
        tsvin.close()


change_to_csv("train.tsv", "train.csv")