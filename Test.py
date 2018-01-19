import pandas as pd
import numpy as np
import csv
import nltk

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


txt = ("This keyboard is in great condition and works like it came out of the box. All of the ports are tested and work perfectly. The lights are customizable via the Razer Synapse app on your PC")

#test stopwoorden
s = set(stopwords.words('english'))
tokens = filter(lambda w: not w in s,txt.split())

append_list = []
for word in tokens:
    lemmatized = WordNetLemmatizer().lemmatize(word,'v')
    append_list.append(lemmatized)

array_list = np.array(append_list)
result = map(str, array_list)
print result

#test words with similar meaning
