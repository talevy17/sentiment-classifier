import pandas as pd
import spacy
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict


class DataCleaner:
    def __init__(self, file):
        self.data = pd.read_csv(file, usecols =["genre","lyrics"])
        print(self.data.shape)
        self.data = self.data.dropna().reset_index(drop=True)
        print(self.data.isnull().sum())


    def cleaner(self, doc):
        txt = [token.lemma_ for token in doc if not token.is_stop]
        if len(txt) > 2:
            return ' '.join(txt)

    def clean_data(self):
        nlp = spacy.load('en', disable=['ner', 'parser'])
        brief = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in self.data['lyrics'])
        txt = [self.cleaner(doc) for doc in nlp.pipe(brief, batch_size=5000, n_threads=-1)]
        clean = pd.DataFrame({'clean': txt})
        return clean.dropna().drop_duplicates()

    def tokenize(self, input):
        data = []
        for i in sent_tokenize(input):
            temp = []
            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())
            print(temp)
            data.append(temp)
        return data