from __future__ import print_function
import gensim, logging, re
from time import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np




BASEPATH = "../train.tsv"
PATH_WORD2VEC_TRAIN = "../traingensim_input"
PATH_WORD2VEC_VALIDATION = "../valgensim_input"

AZ_PATTERN = re.compile(r'[^a-zA-Z\n\r\t\s_]+')

COLUMN_PRICE = 5

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


with open(BASEPATH) as f:
    splitpoint = 0.8*file_len(BASEPATH)
    with open(PATH_WORD2VEC_TRAIN, "w") as tsf:
        with open(PATH_WORD2VEC_VALIDATION, "w") as vsf:
            for i, l in enumerate(f):
                if i < splitpoint:
                    useful = l.split("\t")
                    linewords = useful[1] + " " + useful[4] + ' ' + useful[7]
                    #print(linewords)
                    linewords = AZ_PATTERN.sub(" ", linewords)
                    #print(linewords)
                    last = linewords[-1]
                    linewords = linewords.lower()[:-1] + "," + useful[COLUMN_PRICE] + last
                    #print(linewords)
                    tsf.write(linewords)
                else:
                    useful = l.split("\t")
                    linewords = useful[1] + " " + useful[4] + ' ' + useful[7]
                    linewords = AZ_PATTERN.sub(" ", linewords)
                    linewords = linewords.lower() + "," + useful[COLUMN_PRICE]
                    vsf.write(linewords)



class GensimSents(object):
    def __init__(self, dirname):
        self.fname = dirname

    def __iter__(self):
            for line in open(self.fname):
                #print(line.split())
                #print(line.split(","))
                #print(line.split(",")[0].split())
                #print("end")
                yield line.split(",")[0].split()

def train_gensim_word2vec():
    sentences = GensimSents(PATH_WORD2VEC_TRAIN)
    model = gensim.models.Word2Vec(sentences)
    model.save('train_model')
    del model


doc = ["n", "i", "k", "e", "nike"]

#train_gensim_word2vec()
MODEL = gensim.models.Word2Vec.load('train_model', mmap='r')

words = filter(lambda x: x in MODEL.wv.vocab, doc)

