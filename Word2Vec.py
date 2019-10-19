from Tokenizer import Tokenizer as tk
from SimpleNeuralNet import SimpleNeuralNet as nn
from enum import Enum
import gensim
import random
import warnings
warnings.filterwarnings(action='ignore')

class Constants(Enum):
    """
    Enum of hyper-parameters constants.
    """
    SEM_EVAL = './Dataset/SemEval2015-English-Twitter-Lexicon.txt'
    Epochs = 7


def cross_validataion(train_set, labels):
    cut = 300
    train_data = train_set[: cut]
    train_labels = labels[: cut]
    valid_data = train_set[cut:]
    valid_labels = labels[cut:]
    return train_data, train_labels, valid_data, valid_labels


def split_data(x):
    data = []
    labels = []
    tok = []
    for i in x:
        i = i.replace('\n', '')
        temp = i.split('\t')
        data.append(temp[1])
        labels.append(temp[0])
    tok.append(data)
    return data, labels, tok


def shuffle_data(x):
    random.shuffle(x)
    return x


def main():
    file = open(Constants.SEM_EVAL.value, 'r')
    data = shuffle_data(file.readlines())
    file.close()
    data, labels, tok = split_data(data)
    model = gensim.models.Word2Vec(tok, min_count=1, size=300, window=5, sg=1)
    train_data, train_labels, valid_data, valid_labels = cross_validataion(data, labels)
    net = nn(model)
    net.train(train_data, train_labels)
    net.validate(valid_data, valid_labels)


def first_ass():
    tok = tk('./Dataset/lyrics15LIN.csv', ['english', 'spanish'], '''!()-[]{};:"\,<>./?@#$%^&*_~''')
    model = gensim.models.Word2Vec(tok.tokenize(), min_count=20, size=300, window=5, sg=1)
    print(model.similarity('man', 'something'))
    b = (model['king'] - model['man'] + model['woman'])
    print(model.similar_by_vector(b))
    b = (model['jesus'] + model['cross'])
    print(model.similar_by_vector(b))


if __name__ == "__main__":
    main()