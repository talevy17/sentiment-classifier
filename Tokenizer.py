from nltk.tokenize import sent_tokenize, WhitespaceTokenizer
from nltk.corpus import stopwords


class Tokenizer:
    def __init__(self, file, langs):
        sample = open(file)
        self.data = sample.read().replace('\n', ' ')
        sample.close()
        self.stop = set('')
        for lang in langs:
            self.stop = self.stop | set((set(stopwords.words(lang))))
        self.punct = '''!()-[]{};:"\,<>./?@#$%^&*_~'''

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
                    print(j)
            data.append(temp)
        return data
