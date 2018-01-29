import matplotlib.pyplot as plt

#
# Zorg dat deze identiek zijn aan de files waarin je dit hebt opgeslagen
# globale variablen hebben dezelfde naam als in train_val_plot.py
#

VAL_ERROR_FILE = "./val_error_values.txt"
TRAIN_ERROR_FILE = "./train_error_values.txt"
XF = "./x_values_written.txt"


#Todo: vul dit in om een grafiek op te slaan
SAVE_NAME = "my_neural_network_results.png" #must end in .png


def get_vals_from_file(some_file):
    value_list = []
    with open(some_file, 'r') as f:
        for line in f:
            value_list.append(int(line[:-1]))
    return value_list

def construe_plot():
    plt.figure()
    plt.axis()
    x_vals = get_vals_from_file(XF)
    train_e_vals = get_vals_from_file(TRAIN_ERROR_FILE)
    val_e_vals = get_vals_from_file(VAL_ERROR_FILE)

    # first vals = x, second vals = y on plot
    plt.plot(x_vals, train_e_vals, "r", label="training error")
    plt.plot(x_vals, val_e_vals, "b", label="validation error")

    plt.xlabel("number of batches trained on")
    plt.ylabel("error measure")
    plt.legend()

    fig1 = plt.gcf()
    fig1.savefig(SAVE_NAME, dpi=100)
    plt.show()
