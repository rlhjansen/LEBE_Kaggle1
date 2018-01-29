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

# Todo: worden deze globale variabelen gebruikt in jouw algortime jochem? zo nee: weg
# Todo: instructies voor wat te runnen, welke volgorde welke variabelen aan te passen voor het runnen
PATH_TRAIN = "../train.tsv"
PATH_TEST = "../test.tsv"

BATCH_SIZE = 1000  # Don't make this number much larger than 100000
DATA_SIZE = sum(1 for line in open(PATH_TRAIN))  # Number of \n in train file
TEST_DATA_SIZE = sum(1 for line in open(PATH_TRAIN)) #Number of \n in train file
THRESH = 100  # How often needs a word to occur before it is used
TRAIN_RATIO = 1.0  # The ratio train validation

# The following constants are the columns in each "data" variable

COLUMN_LABEL = 100

TEST_COLUMN_ID = 0
TEST_COLUMN_NAME = 1
TEST_COLUMN_CONDITION = 2
TEST_COLUMN_CATEGORY = 3
TEST_COLUMN_BRAND = 4
TEST_COLUMN_SHIPPING = 5
TEST_COLUMN_DESCRIPTION = 6


# These characters will be ignored by the neural net when training data
IGN_CHAR = [',', ':', ';', '.', '(', ')', '\'', '"', '!', '?', '*', '&', '^']


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


# files waarin error na X hoeveelheid batches word opgeslagen
# hieruit kan met de file "make_error_graph.py" een grafiek gemaakt worden
VAL_ERROR_FILE = "./val_error_values.txt"
TRAIN_ERROR_FILE = "./train_error_values.txt"
XF = "./x_values_written.txt"



#Todo: waarden hieronder invullen, dit in verband met hoe de delta functies werken. ik weet dat niet, jochem wel.

TRAIN_SIZE = "moet bepaald"
MAX_TRIES = "een integer, hoeveel keer er uberhaubt maximaal over de train data heen gegaan word"

VAL_INPUT_PATH = "" #invullen waar de validation data staat die kan worden ingeladen
TRAIN_INPUT_PATH = "" #invullen waar de train data staat die kan worden ingeladen

LAYERS = 10
SAVED_NN = "NN_pickle_" + str(LAYERS) + "_layers_.pkl"
ERROR_INTERVAL = 10


E_VAL_REGR_TESTING_PATH = "" # pad waaruit data geladen word om de regressor te testen. (validation error)
E_TRAIN_REGR_TESTING_PATH = "" # pad waaruit data geladen word om de regressor te testen. (training error)

# Note: Ik heb de batches nu in een random volgorde laten gaan, wat voor
#       verbetering zou moeten zorgen.

def save_stuff(regr, cur_x):
    joblib.dump(regr, SAVED_NN)
    write_value(test_regressor(regr, E_VAL_REGR_TESTING_PATH), VAL_ERROR_FILE)
    write_value(test_regressor(regr, E_TRAIN_REGR_TESTING_PATH), TRAIN_ERROR_FILE)
    write_value(cur_x, XF)

def write_value(val, file):
    with open(file, "a") as f:
        f.write(str(val) + "\n")


def test_regressor(regr, path):
    """Return the average squared error of the regression neural network"""
    stop = False
    err_val = 0.0
    label_len = 0.0
    while not stop:
        data, labels, stop = "load_func with VAL or TRAIN PATH (should be same format)" #load_some_data(path, start, batch_size)
        pred = regr.predict(data)
        err_val += np.sum(np.power(np.log(pred+1) - np.log(labels+1), 2))
        label_len += float(len(labels))
    return sqrt(err_val/label_len)


#########################################################
#                                                       #
#               main function below here                #
#                                                       #
#########################################################

def main(layers, startfile=None):
    """The main function of the program"""
    num_batches = int(TRAIN_SIZE / BATCH_SIZE)
    print("The program will run in", num_batches)

    i = 0
    if startfile: #pickle file
        regr = joblib.load(SAVED_NN)
    else:
        regr = MLPRegressor(hidden_layer_sizes=layers)

    x_val = 0
    for _ in range(MAX_TRIES):
        shuffled_batches = range(num_batches)
        shuffle(shuffled_batches)
        batches_passed = 0
        for batch in shuffled_batches:
            print("\nCurrent loop =", i + 1)
            print("Current layers:", layers)
            print("Starting batch", batches_passed + 1, "of", num_batches)

            print("Loading data...")

            #Todo: deze laadfunctie moet dus geschreven worden (waarmee ik bedoel dat je de delta load (of wat voor naam je het ook geeft hier moet verwerken, de traindata en train labels die hieruit komen moeten geschrikt zijn om in de partial fit te stoppen)
            train_data, train_labels, stop = load_train(PATH_INPUT_TRAIN, batch * BATCH_SIZE, BATCH_SIZE)

            # train_data, train_labels = load_simple_train(PATH_DELTA_TRAIN,batch * BATCH_SIZE,BATCH_SIZE)

            print("Training neural network...")
            regr.partial_fit(train_data, train_labels)
            batches_passed += 1
            i += 1
            if batch % ERROR_INTERVAL == 0:
                print("don't interrupt untill you see: Current loop= X")
                x_val += ERROR_INTERVAL
                save_stuff(regr, x_val)



TRAINLAYERS = [100]*LAYERS

# main(TRAINLAYERS, startfile=SAVED_NN) # If starting warm. Else use next line
# main(TRAINLAYERS)
