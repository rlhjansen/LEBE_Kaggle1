

completefile = "../input_train_100.tsv"
train_write = "../train_part_100.tsv"
val_write = "../val_part_100.tsv"

train_test_write = "../train_test_100.tsv"
val_test_write = "../val_test_100.tsv"


file_length_complete = sum(1 for line in open(completefile))
ratio = 0.8
cutoff_point = float(file_length_complete)*ratio

with open(completefile, 'r') as cf:
    with open(train_write, 'w') as tw:
        with open(val_write, 'w') as vw:
            for i in range(file_length_complete):
                if i < cutoff_point:
                    tw.write(cf.readline())
                else:
                    vw.write(cf.readline())

with open(train_write, 'r') as tw:
    with open(train_test_write, 'w') as ttw:
        for i in range(10000):
            ttw.write(tw.readline())

with open(val_write, 'r') as vw:
    with open(val_test_write, 'w') as vtw:
        for i in range(10000):
            vtw.write(vw.readline())
