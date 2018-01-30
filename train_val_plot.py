""" This program trains a regressor on the dataset. Running MLPRegressor
requires a sklearn version of 0.18 or higher.
    This program expects a file that cPickle can open at location PATH_TRAIN
cPickle needs to load a 2 dimensional numpy array, each row is a datapoint and
the last element of each row should be the label."""
# essential imports
from __future__ import print_function
from sklearn.neural_network import MLPRegressor
import numpy as np
from math import sqrt
from sklearn.externals import joblib
from random import shuffle

# Input files and it's sizes
VEC_LEN = 100  # Determines which train and val data is used
PATH_TRAIN = "../delta_train_" + str(VEC_LEN) + ".tsv"
PATH_VAL = "../delta_val_" + str(VEC_LEN) + ".tsv"
PATH_TRAIN_TST = "../delta_train_tst_" + str(VEC_LEN) + ".tsv"
TRAIN_SIZE = sum(1 for line in open(PATH_TRAIN))  # Number of \n in train file
VAL_SIZE = sum(1 for line in open(PATH_VAL)) #Number of \n in train file

# Output files
LAYERS = 10  # Determines which data is loaded
VAL_ERROR_FILE = "../val_error_values.txt"
TRAIN_ERROR_FILE = "../train_error_values.txt"
XF = "../x_values_written.txt"
SAVED_NN = "../NN_pickle_" + str(LAYERS) + "_layers_.pkl"

# Important parameters on which the program is run
BATCH_SIZE = 1000  # Don't make this number much larger than 100000
TRAIN_TST_SIZE = 10000  # How many rows do we test when calculating error
THRESH = 100  # How often needs a word to occur before it is used
MAX_TRIES = 100
ERROR_INTERVAL = 10


#########################################################
#                                                       #
#   Block after which comes new code for saving stuff   #
#                                                       #
#########################################################

#
# Belangrijke informatie:
# test_regressor herschreven.
#
# Todo: voeg load functies aan toe voor main() en test_regressor()
#

# Note: Ik heb de batches nu in een random volgorde laten gaan, wat voor
#       verbetering zou moeten zorgen.

def load_data(path, start, size):
    """Return data as an array."""
    data = []
    labels = []
    stop = False

    with open(path) as f:

        i = 0
        for row in f:
            if i < start:
                i += 1
                continue
            elif i >= start + size:
                break

            row = row[:-1]
            row = row.split("\t")
            row = row[1:]
            label = float(row[-1])
            row = row[:-1]

            labels.append(label)
            data.append(np.array(row, dtype="float"))
            i += 1

        if i <= start + size:
            stop = True

    return np.array(data), np.array(labels), stop


def make_train_test(path):
    """Make a file with random rows from the training set"""
    with open(path, 'a') as f_a:
        with open(PATH_TRAIN) as f_r:
            i = 0
            for row in f_r:
                f_a.write(row)
                i += 1
                if i >= TRAIN_TST_SIZE:
                    break


def save_stuff(regr, cur_x):
    """Save the progress of the regressor"""
    joblib.dump(regr, SAVED_NN)
    write_value(test_regressor(regr, PATH_TRAIN_TST), VAL_ERROR_FILE)
    write_value(test_regressor(regr, PATH_VAL), TRAIN_ERROR_FILE)
    write_value(cur_x, XF)


def write_value(val, path):
    """Append a single line in a file given by path"""
    with open(path, "a") as f:
        f.write(str(val) + "\n")


# Note: This function now tests through the whole train or validation file.
# We might want to specify a max size or something.
def test_regressor(regr, path):
    """Return the average squared error of the regression neural network"""
    pointer = 0
    stop = False
    err_val = 0.0
    label_len = 0.0
    while not stop:
        data, labels, stop = load_data(path, pointer, BATCH_SIZE)
        pred = regr.predict(data)
        err_val += np.sum(np.power(np.log(pred+1) - np.log(labels+1), 2))
        label_len += float(len(labels))
        pointer += BATCH_SIZE
    return sqrt(err_val/label_len)


#########################################################
#                                                       #
#               main function below here                #
#                                                       #
#########################################################

def main(layers, startfile=None):
    """The main function of the program"""
    print("Making a test file for the training set.")
    make_train_test(PATH_TRAIN_TST)

    num_batches = int(TRAIN_SIZE / BATCH_SIZE)
    print("The program will run in", num_batches)

    i = 0
    if startfile: #pickle file
        regr = joblib.load(SAVED_NN)
    else:
        regr = MLPRegressor(hidden_layer_sizes=layers)

    x_val = 0
    for i in range(MAX_TRIES):
        shuffled_batches = range(num_batches)
        shuffle(shuffled_batches)
        batches_passed = 0
        for batch in shuffled_batches:
            print("\nCurrent loop =", i + 1)
            print("Current layers:", layers)
            print("Loading batch", batches_passed + 1, "of", str(num_batches) + "...")

            #Todo: deze laadfunctie moet dus geschreven worden (waarmee ik bedoel dat je de delta load (of wat voor naam je het ook geeft hier moet verwerken, de traindata en train labels die hieruit komen moeten geschrikt zijn om in de partial fit te stoppen)
            train, labels, _ = load_data(PATH_TRAIN, batch * BATCH_SIZE, BATCH_SIZE)

            print("...Training neural network...")
            regr.partial_fit(train, labels)
            batches_passed += 1
            if batches_passed % ERROR_INTERVAL == 0:
                print("...Saving and testing, don't interrupt the program...")
                x_val += ERROR_INTERVAL
                save_stuff(regr, x_val)
                print("...Done")




TRAINLAYERS = [100]*LAYERS

# main(TRAINLAYERS, startfile=SAVED_NN) # If starting warm. Else use next line
main(TRAINLAYERS)
