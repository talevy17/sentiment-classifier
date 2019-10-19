# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from nltk.corpus import stopwords
import string
import nltk
warnings.filterwarnings(action='ignore')
import gensim

#  Reads ‘alice.txt’ file
sample = open("./Dataset/lyrics15LIN.csv", "r")
f = sample.read()

# Replaces escape character with space
stop = set(stopwords.words('english'))
stop = stop | (set(stopwords.words('spanish')))
# freq = nltk.FreqDist(f)
# stop = stop | set(freq.keys())
data = []
# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        if(j.isalpha() and (j.lower() not in stop) and len(j) > 1):
            temp.append(j.lower())
    print(temp)
    data.append(temp)
# Create Skip Gram model

model = gensim.models.Word2Vec(data, min_count=1, size=300,
                                window=5, sg=1)
print(model.similarity('distancia', 'something'))

b = (model['king'] - model['man'] + model['woman'])
print(model.similar_by_vector(b))
b = (model['man'] - model['money'] - model['home'])
print(model.similar_by_vector(b))