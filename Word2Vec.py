import nltk
from nltk.corpus import stopwords

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
from collections import defaultdict

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
    network = Fc()
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

def printDictAfterFC(dict):
    for i in sorted(dict.values(),reverse = True):
        for j in dict.keys():
            if dict.get(j) == i:
                print(j, ":", i)

def load3000(model, path, file, name):

    tok = dc(path, ['english', 'spanish'], '''!()-[]{};:"\,<>./?@#$%^&*_~''')
    data = file['lyrics']
    sentences = []
    punct = '''!()-[]{};:"\,<>./?@#$%^&*_~'''

    # sentences_t =[]
    # for song_lyrics in data:
    #     sentences_t += tok.tokenize_sentences(song_lyrics)
    # with open("Dataset/{}".format(name),'w') as f:
    #     for s in sentences_t:
    #         for i in s:
    #             f.write(str(i) + "\n")

    # word_freq = defaultdict(int)
    #
    # with open("./Dataset/" + name,'r') as f:
    #     for w in f:
    #             w = w.strip()
    #             sentences.append(w)
    #             word_freq[w]+=1

    # flat_list = [item for sublist in songs for item in sublist]
    # word_count = sorted(word_freq, key=word_freq.get, reverse=True)[:3000]
    # words = [w for w in word_count if not w in set(stopwords.words("english")) and len(w)>3]
    # with open("Dataset/3000_Words",'w') as f:
    #     for s in words:
    #             f.write(str(s) + "\n")
    words_final =[]
    with open("./Dataset/3000_Words",'r') as f:
        for s in f:
            words_final.append(s.strip())
    print(words_final)
    return words_final

# def makeList50PerGenre(words_3000, path):
#     tok = dc(path, ['english', 'spanish'], '''!()-[]{};:"\,<>./?@#$%^&*_~''')
#     final_list = []
#     data = pd.read_csv(path)
#     data = pd.DataFrame (data, columns=['genre','lyrics'])
#     grouped = data.groupby('genre')
#     genres = data['genre'].items()
#     songs = data['lyrics'].items()
#     #data_songs = pd.read_csv('path')
#     dict = {}
#     for (song,genre) in zip(songs,genres):
#         print(str(genre) + ' : ' + str(song))
#         for word in song:
#             if genre in dict[word]
#         word_freq = defaultdict(int)
#         words = []
#         words += tok.tokenize_sentences(str(song))
#         for w in words:
#             for i in w:
#                 if words_3000.count(i) >0:
#                     word_freq[i] +=1
#     final_list+= sorted(word_freq, key=word_freq.get, reverse=True)[:50]
    with open("Dataset/50_Words_per_genre",'w') as f:
        for s in final_list:
                f.write(str(s) + "\n")
    words_final =[]
    with open("./Dataset/50_Words_per_genre",'r') as f:
        for s in f:
            words_final.append(s.strip())
    print(words_final)
    print(len(words_final))
    return words_final

def dictGenreLyrics(path):
    dict ={}
    data =pd.read_csv(path)
    tok = dc(path, ['english', 'spanish'], '''!()-[]{};:"\,<>./?@#$%^&*_~''')
    genres = data[data['genre'].notnull()]
    print (genres)


    for i in range(len(data)):
        lyrics = data.get_value(i,5,takeable= True)
        genre = data.get_value(i,4,takeable= True)
        lir_parsed = tok.tokenize_sentences(lyrics)
        dict[genre] += lir_parsed

    f = open('Dataset/Dictionary_genre_lyrics.txt','w')
    f.write(str(dict))
    f.close()

    f = open('./Dataset/Dictionary_genre_lyrics.txt','r')
    loadDic=f.read()
    f.close()
    return loadDic


def main():
    path = './Dataset/lyrics.csv'
    fileSongs,tokenizer = load(path)
    dataSongs = fileSongs['lyrics']
    name = "LIN380new"
    #buildModelWordToVec(path,data,name)
    toOpen = "./Dataset/" + name
    model = Word2Vec.load(toOpen)
    #similarities(model)
    #VectorsAlgebra(model)
    #distances(model, 'woman','girl')
    fileSEM = open(Constants.SEM_EVAL.value, 'r')
    #data = shuffle_data(file.readlines())
    dataSEM = (fileSEM.readlines())

    fileSEM.close()
    dataSEM, labelsSEM, Semtok = split_data(dataSEM)
    #word2vec(Semtok, 'SemEVAL', 300, 1, 4,10,1e-3)
    #modelSem = Word2Vec.load('./Dataset/SemEVAL')
    network = Fc()
    #dict =fully_connected(model,modelSem,dataSEM,labelsSEM,network)
    #printDictAfterFC(dict)
    words_final3000 = load3000(model,path,fileSongs, 'sentences')
    #makeList50PerGenre(words_final3000,path)
    dictGenreLyrics(path)

if __name__ == "__main__":
    main()