import pandas as pd
import spacy
import re
import nltk
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer
from nltk.corpus import stopwords

from collections import defaultdict


class DataCleaner:
    def __init__(self, data, langs , punctuation):
        self.stop = set('')
        self.data = data
        for lang in langs:
            self.stop = self.stop | (set(stopwords.words(lang)))
        self.stop = self.stop | set(punctuation)
        self.punc = punctuation

    def regex(self, word):
        for char in word:
            if char in self.punct:
                return False
        return True

    def cleaner(self, doc):
        txt = [token.lemma_ for token in doc if not token.is_stop]
        if len(txt) > 2:
            return ' '.join(txt)

    def clean_data_words(self, text):
        text = re.sub("[^a-zA-Z]", " ",text)
        words = text.lower().split()
        words = [w for w in words if not (w.lower() in self.stop and len(w)<3)]
        return  words
       ## words = text.lower().split()
        #self.stop = self.stop | set(stopwords.words("english"))
        ##words = [w for w in words if not (w.lower() in self.stop and self.regex(w))]
        ##return words
        #brief1 = (re.sub("[^a-zA-Z']", ' ', str(row)).lower() for row in self.data_lyrics)
        #txt1 = [self.cleaner(doc) for doc in nlp.pipe(brief, batch_size=5000, n_threads=-1)]
        #clean1 = pd.DataFrame({'': txt})
        #return clean.dropna().drop_duplicates()

    def tokenize_sentences(self,song):
        input = song.strip('()')
        #input = input.replace('\\n', " ")
        #input = re.sub('[0-9]', "",input)
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(input) if tokenizer else input
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 1:
                data = self.clean_data_words(raw_sentence)
                if len(data)>0:
                    sentences.append(data)
        return sentences





