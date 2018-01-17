# This is a stub file, a file in which all (important) functions are explained.


# num_data: n x 2 float array; 1e colom condition, 2e colom shipping
# words: n x 1 string array; elke rij: woorden die voorkomen in die rij
# labels: n x 1 float array: prijzen van de rij
# word_map: dict() die woorden mapt met ints
def data_to_input(num_data, words, labels, word_map):
    """Return a version of the data that a neural network can take as input."""
    in_vecs = []
    n_words = len(word_map.keys())
    for i, word_row in enumerate(words):
        in_row = np.zeros(n_words + len(num_data[0]) + 1)  # This becomes a row for "in_vecs"
        for word in word_row:
            if word in word_map:
                in_row[word_map[word]] = 1.0
        for j, _ in enumerate(num_data[i]):
            in_row[-(j + 2)] = num_data[i][j] 
        in_row[-1] = float(labels[i])
        in_vecs.append(in_row)
    return np.array(in_vecs)
# in_vecs: [[word word word ... word shipping condition price] ... ]


# start: int; How many lines to skip before reading the data file.
# size: int; How many lines to read.
def load_data(start, size):
    """Return the trainingdata as an array."""
    data = []
    labels = []
    if start + size > DATA_SIZE:
        size = DATA_SIZE - start

    with open(PATH_TRAIN) as f:
        for _ in range(start):
            f.readline()
        for _ in range(size):
            line = f.readline()
            line = line[:-1]  # Remove newline
            line = line.split("\t")
            label = line[COLUMN_LABEL]
            line = line[1:COLUMN_LABEL] + line[COLUMN_LABEL+1:]
            data.append(np.array(line))
            labels.append(label)

    return np.array(data), np.array(labels)
# data: n x 6 array; 
# label: n x 1 array; prices


# start: int; How many lines to skip before reading the data file.
# size: int; How many lines to read.
def load_input(start, size):
    """Return input vectors as an 2-dim array."""
    pass
# data
# label: n x 1 array; prices


# This is the main function of the preprocessing.py
def main():
    """The main function of the program"""
    pointer = 0
    batch_count = 0

    print("Batch size =", BATCH_SIZE, "Data size =", DATA_SIZE)

    data, labels = load_data(0, 1)
    while pointer < DATA_SIZE:
        print("Now starting batch", batch_count, "...")
        new_data, new_labels = load_data(pointer, BATCH_SIZE)
        data = np.concatenate((data, new_data), axis=0)
        labels = np.concatenate((labels, new_labels), axis=0)
        pointer += BATCH_SIZE
        batch_count += 1
    data = data[1:]
    labels = labels[1:]

    words = sentence_to_words(data)  # process name, brand, description
    num_data = numeric_data(data)  # process condition, shipping
    word_count = count_words(words)  # return a counter of all words
    word_map = map_words(words, word_count)  # return a map[word] = int
    in_vecs = data_to_input(num_data, words, labels, word_map)
    store(in_vecs, PATH_INPUT_VECTOR)  # ToDo


# arr: any 2-dim array
# path: string; path to storage file
# pointer: int; from which byte in the file do you start storing the "arr"
def store(arr, path, pointer):
    pass
# pointer: int; at which byte did you stop storing
# how to store: word word word ... word shipping condition price\n ... (use tabs instead of spaced)
