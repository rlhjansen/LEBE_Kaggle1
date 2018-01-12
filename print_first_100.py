import csv
import os
import re

s = ''.join(chr(i) for i in range(256)) # all possible bytes
name_discr_pattern = re.compile(r'[^\s\w_]+') # What will remain
number_pattern = re.compile(r'[^\s\w_.]+') # What will remain
category_pattern = re.compile(r'[^\s\w_./\'&,]+') # What will remain
strippedList = name_discr_pattern.sub(' ', s)
strippedList.strip()


print(s)
print(strippedList)



def printrows(infile, N):
    with open(os.path.join(os.pardir, infile)) as tsvin:
        tsvs = csv.reader(tsvin, delimiter='\t')
        i = 0
        for row in tsvs:
            regexrow = [number_pattern.sub('', row[0]), name_discr_pattern.sub('', row[1]), number_pattern.sub('', row[2]), category_pattern.sub('', row[3]), name_discr_pattern.sub('', row[4]), name_discr_pattern.sub('', row[5]), row[6], number_pattern.sub('',row[7])]
            print([j.strip() for j in regexrow])

            print(row)
            i += 1
            if i == N:
                break
        tsvin.close()

printrows("trainColumnSwitched.tsv", 100)
