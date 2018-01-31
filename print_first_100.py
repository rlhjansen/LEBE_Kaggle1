import csv
import os

completefile = "../input_train_100.tsv"
train_write = "../train_part_100.tsv"
train_test_write = "../train_test_100.tsv"

val_write = "./.val_part_100.tsv"
val_test_write = "../val_test_100.tsv"



def printrows(infile, N):
    with open(infile) as tsvin:
        tsvs = csv.reader(tsvin, delimiter='\t')
        i = 0
        for row in tsvs:
            print(row)
            i += 1
            if i == N:
                break
        tsvin.close()
        print("startnext")

printrows(completefile, 10)
printrows(val_write, 10)
printrows(val_test_write, 10)