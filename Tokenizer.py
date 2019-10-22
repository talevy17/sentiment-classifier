from nltk.tokenize import sent_tokenize, WhitespaceTokenizer
from nltk.corpus import stopwords
import re


class Tokenizer:
    def __init__(self, file, langs, punctuation):
        self.data = file
        self.data = self.data.to_string()
        self.data = self.data.replace('\\n', " ")
        self.data = re.sub('[0-9]', " ",self.data)
        self.stop = set('')
        for lang in langs:
            self.stop = self.stop | set((set(stopwords.words(lang))))
        self.punct = punctuation

    def regex(self, word):
        for char in word:
            if char in self.punct:
                return False
        return True

    def tokenize(self):
        data = []
        for i in sent_tokenize(self.data):
            temp = []
            for j in WhitespaceTokenizer().tokenize(i):
                if self.regex(j) and (j.lower() not in self.stop) and len(j) > 1:
                    temp.append(j.lower())
            if len(temp)>0:
             data.append(temp)
        return data
