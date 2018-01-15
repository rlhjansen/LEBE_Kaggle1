import os, csv, pickle

# we open the file for reading


#
# prints column ordering for training & testset input files
#

"""

with open(os.path.join(os.pardir, "train.tsv"),
          encoding="utf8") as traindatatsv:
    traindata = csv.reader(traindatatsv, delimiter='\t')
    for row in traindata:
        print(row)
        break
    traindatatsv.close()
with open(os.path.join(os.pardir, "test.tsv"), encoding="utf8") as testdatatsv:
    testdata = csv.reader(testdatatsv, delimiter='\t')
    for row in testdata:
        print(row)
        break
    testdatatsv.close()

#"""




#
# creates a copy of the training data where price& description are switched
#
with open(os.path.join(os.pardir, "train.tsv"), encoding="utf8") as traindatatsv:
    traindata = csv.reader(traindatatsv, delimiter='\t')
    with open(os.path.join(os.pardir, "trainColumnSwitched.tsv"), "w", encoding="utf8") as trainswitchtsv:
        writer = csv.writer(trainswitchtsv, delimiter='\t', skipinitialspace=True)
        for row in traindata:
            writer.writerow([row[0], row[1], row[2], row[3], row[4], row[7], row[6], row[5]])
        trainswitchtsv.close()
    traindatatsv.close()



#
# creates a copy of the test data where shipping & description are switched
# using both of these the training & test data are ordered the same way
#

#"""
with open(os.path.join(os.pardir, "test.tsv")) as testdatatsv:
    testdata = csv.reader(testdatatsv, delimiter='\t')
    with open(os.path.join(os.pardir, "testColumnSwitched.tsv"), "w") as testswitchtsv:
        writer = csv.writer(testswitchtsv, delimiter='\t', skipinitialspace=True)
        for row in testdata:
            writer.writerow([row[0], row[1], row[2], row[3], row[4], row[6], row[5]])
        testswitchtsv.close()
    testdatatsv.close()
#"""
