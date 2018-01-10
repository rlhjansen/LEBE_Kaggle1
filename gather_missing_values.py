import csv
import os

with open(os.path.join(os.pardir, "train.tsv"), encoding="utf8") as tsvin:
    with open(os.path.join(os.pardir, "missval.tsv"), 'w') as missout:
        tsvs = csv.reader(tsvin, delimiter='\t')
        i = -1
        for row in tsvs:
            indices = [str(i) for i, x in enumerate(row) if x == ""]
            if len(indices)>0:
                indwrite = str(i) + "\t" + ",".join(indices) + '\n'
                missout.write(indwrite)
            i += 1
        missout.close()
    tsvin.close()

with open(os.path.join(os.pardir, "missval.tsv")) as missout:
    tsvs = csv.reader(missout, delimiter='\t')
    indices = [0]*8
    for row in tsvs:
        l = [int(i) for i in row[1].split(",")]
        for i in l:
            indices[i] += 1
    with open(os.path.join(os.pardir, "missindeces.txt"), 'w') as out:
        indices = [str(i) for i in indices]
        out.write(" ".join(indices))
    out.close()
missout.close()


