import nltk

from Tokenizer import Tokenizer as tk
from DataCleaner import DataCleaner as dc
from SimpleNeuralNet import SimpleNeuralNet
from enum import Enum
import numpy
import gensim
import random
import warnings
import pandas as pd
import logging
from gensim.models import Word2Vec
import re
from FullyConnected import FullyConnected as Fc

from sklearn.linear_model import LinearRegression

warnings.filterwarnings(action='ignore')

class Constants(Enum):
    """
    Enum of hyper-parameters constants.
    """
    SEM_EVAL = './Dataset/SemEval2015-English-Twitter-Lexicon.txt'
    Epochs = 7


# def train(fc, opt, word2vec, data, label_set):
#     loss_function = nn.MSELoss()
#     labels = numpy.asarray(label_set)
#     for epoch in range(1):
#         fc.train()
#         # print('Epoch: ' + str(epoch))
#         for k, (word, label) in enumerate(zip(data, labels)):
#             temp = torch.from_numpy(word2vec[word])
#             temp = temp.reshape(-1, temp.size(0))
#             prediction = fc(temp)
#             loss = loss_function(prediction, torch.tensor(label))
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         # validation(conv, valid_set, device)
#     return fc
#
#
# def validate(net, word2vec, data, labels):
#     net.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for word, label in zip(data, labels):
#             # calc on device (CPU or GPU)
#             # calc the prediction vector using the model's forward pass.
#             temp = torch.from_numpy(word2vec[word])
#             temp = temp.reshape(-1, temp.size(0))
#             pred = net(temp)
#             print(pred)
#             total += 1
#             if pred.data * torch.tensor(label).data > 0:
#                 correct += 1
#         # print the accuracy of the model.
#         print('Test Accuracy of the model: {}%'.format((correct / total) * 100))
#
#
# def test(net, word2vec, test):
#     net.eval()
#     with torch.no_grad():
#         for word in test:
#             # calc on device (CPU or GPU)
#             # calc the prediction vector using the model's forward pass.
#             pred = net(word2vec[word])
#             print(word+' '+pred)


def cross_validation(train_set, labels):
    cut = 1100
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
        labels.append(numpy.asarray(float(temp[0])))
    tok.append(data)
    return data, labels, tok


def shuffle_data(x):
    random.shuffle(x)
    return x


def aaa():
    file = open(Constants.SEM_EVAL.value, 'r')
    data = shuffle_data(file.readlines())
    file.close()
    data, labels, Semtok = split_data(data)
    model = gensim.models.Word2Vec(min_count=20, size=300, window=5, sg=1, iter=1)
    model.build_vocab(Semtok)
    toke = tk('./Dataset/lyrics15LIN.csv', ['english', 'spanish'], '''!()-[]{};:"\,<>./?@#$%^&*_~''')
    tokenSongs = toke.tokenize()
    model_test = gensim.models.Word2Vec(min_count=20, size=300, window=5, sg=1, iter=1)
    model_test.build_vocab(tokenSongs)
    model_test.train(tokenSongs,total_examples=model.corpus_count,epochs=1)

    # train_data, train_labels, valid_data, valid_labels = cross_validation(data, labels)
    fc = SimpleNeuralNet(model)
    # print(len(train_labels))
    fc.train(data, labels,network)
    fc.test(Semtok, model_test, network)
    #fc.validate(valid_data, valid_labels)
    # net.train(data, labels)
    # net.validate(data, labels)
def load(path):
        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
        file = pd.read_csv(path)
        file = file[file['lyrics'].notnull()]
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return file,tokenizer
def word2vec(sentences, name, num_features=300, min_word_count=40, workers=4,
             context=10, downsampling=1e-3):
    model = Word2Vec(sentences,
                     workers=workers,
                     size=num_features,
                     min_count=min_word_count,
                     window=context,
                     sample=downsampling)
    model.init_sims(replace=True)
    model_name = "Dataset/{}".format(name)
    model.save(model_name)
    return model

def buildModelWordToVec(path,data,name):
    tok = dc(path, ['english', 'spanish'], '''!()-[]{};:"\,<>./?@#$%^&*_~''')
    sentences =[]
    for song_lyrics in data:
        sentences += tok.tokenize_sentences(song_lyrics)
    word2vec(sentences, name)
def similarities(model):
    print(model.wv.most_similar('house'))
    print(model.wv.most_similar('king'))
    print(model.wv.most_similar('israel'))
    print(model.wv.most_similar('jesus'))
    print(model.wv.most_similar('mother'))
    print(model.wv.most_similar('god'))

def VectorsAlgebra(model):
    print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))
    print(model.wv.most_similar(positive=['jewish', 'king'], negative=['jesus']))
    print(model.wv.most_similar(positive=['spider', 'pig'], negative=['web']))
    print(model.similar_by_vector(model['king'] + model['woman'] - model['man']))
    print(model.similar_by_vector(model['spider'] + model['pig']))

def distances(model, w1,w2):
    cd = model.wv.similarity(w1, w2)
    print("Cosine Distance = ", cd)

    ed = numpy.linalg.norm(model[w1] - model[w2])
    print("Euclidean Distance = ", ed)
def fully_connected(model,modelSEM,data, labels,network):
    fc = SimpleNeuralNet(modelSEM)
    fc.train(data, labels,network)
    dict  = fc.test(model.wv.vocab,model,network)
    return dict


def main():
    path = './Dataset/lyrics.csv'
    file,tokenizer = load(path)
    data = file['lyrics']
    name = "LIN380new"
    #buildModelWordToVec(path,data,name)
    toOpen = "./Dataset/" + name
    model = Word2Vec.load(toOpen)
    #similarities(model)
    #VectorsAlgebra(model)
    #distances(model, 'woman','girl')
    file = open(Constants.SEM_EVAL.value, 'r')
    #data = shuffle_data(file.readlines())
    data = (file.readlines())

    file.close()
    data, labels, Semtok = split_data(data)
    #word2vec(Semtok, 'SemEVAL', 300, 1, 4,10,1e-3)
    modelSem = Word2Vec.load('./Dataset/SemEVAL')
    network = Fc()
    dict =fully_connected(model,modelSem,data,labels,network)


    for i in sorted(dict.values(),reverse = True):
        for j in dict.keys():
            if dict.get(j) == i:
                print(j, ":", i)


if __name__ == "__main__":
    main()