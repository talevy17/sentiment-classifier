# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from nltk.corpus import stopwords
import string
import nltk


warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

#  Reads ‘alice.txt’ file
sample = open("./Dataset/lyrics1.csv", "r")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")
stop = set(stopwords.words('english'))
stop = stop | (set(stopwords.words('spanish')))
freq = nltk.FreqDist(f)
stop = stop | set(freq.keys())

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        if(j.isalpha() and (j.lower() not in stop) and len(j) > 1 ):
         temp.append(j.lower())
    data.append(temp)
# Create Skip Gram model

model2 = gensim.models.Word2Vec(data, min_count=1, size=300,
                                window=5, sg=1)
print(model2.similarity('distancia', 'funciona'))
