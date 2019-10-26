import csv
import pickle

import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

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
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

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
    # stop = set(stopwords.words("english"))
    # with open("./Dataset/" + name,'r') as f:
    #     for w in f:
    #             w = w.strip()
    #             if not w in stop and len(w)>2:
    #                 sentences.append(w)
    #                 word_freq[w]+=1
    # word_freq = sorted(word_freq, key=word_freq.get, reverse=True)[:3000]
    # with open("Dataset/3000_Words",'w') as f:
    #     for s in word_freq:
    #             f.write(str(s) + "\n")
    words_final =[]
    with open("./Dataset/3000_Words",'r') as f:
        for s in f:
            words_final.append(s.strip())
    print(words_final)
    return words_final


def words_3000_print(word_3000, dic_gen_lyrics):
    dict = dic_gen_lyrics
    dic_most = {}
    for w in word_3000:
        dic_most[w] = w + " : "
    for genre in dict.keys():
        word_freq = defaultdict(int)
        for sentence in dict[genre]:
            for word in sentence:
                word_freq[word] +=1
        for w in dic_most.keys():
            dic_most[w] +=genre +" " + str(word_freq[w])+ " "
    for i in dic_most.values():
        print(i)
    return dic_most




def makeList50PerGenre(words_3000, dic_gen_songs ,path):
    # dic_50 ={}
    # stop = set(stopwords.words("english"))
    # dict = dic_gen_songs
    # for genre in dict.keys():
    #     word_freq = defaultdict(int)
    #     for sent in dict[genre]:
    #         for word in sent:
    #             if not word in stop and len(word)>2:
    #                 word_freq[word] += 1
    #     dic_50[genre]= sorted(word_freq, key=word_freq.get, reverse=True)[:50]
    # output = open('Dataset/Dict_50.pkl', 'wb')
    # pickle.dump(dic_50, output)
    # output.close()

    pkl_file = open('./Dataset/Dict_50.pkl', 'rb')
    loadDic = pickle.load(pkl_file)
    pkl_file.close()
    return loadDic




def dictGenreLyrics(path):
    dict ={}
    data =pd.read_csv(path)
    tok = dc(path, ['english', 'spanish'], '''!()-[]{};:"\,<>./?@#$%^&*_~''')
    genres = data['genre'].dropna().drop_duplicates().tolist()
    # for g in genres:
    #     dict[g] = []
    # for i in range(len(data)):
    #     lyrics = data.get_value(i,5,takeable= True)
    #     if not pd.isnull(lyrics):
    #         genre = data.get_value(i,4,takeable= True)
    #         lir_parsed = tok.tokenize_sentences(lyrics)
    #         dict[genre] += lir_parsed
    # output = open('Dataset/Dict_gen_songs.pkl', 'wb')
    # pickle.dump(dict, output)
    # output.close()

    pkl_file = open('./Dataset/Dict_gen_songs.pkl', 'rb')
    loadDic = pickle.load(pkl_file)
    pkl_file.close()

    return loadDic

def display_2D_results(w2v_model, most_frequent_words_by_genre):
    number_of_colors = len(most_frequent_words_by_genre)

    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    print(color)

    color_list = []
    arrays = np.zeros((0, 300), dtype='f')
    for counter, (genre, words) in enumerate(most_frequent_words_by_genre.items()):
        for word in words:
            try:
                vec_word = [w2v_model[word]]
                color_list.append(color[counter])
                arrays = np.append(arrays, vec_word, axis=0)
            except:
                print("'", word, "'", "does not appear in the model")

    reduce = PCA(n_components=50).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduce)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", scatter_kws={'facecolors': df['color']})

    plt.xlim(Y[:, 0].min() - 10, Y[:, 0].max() + 10)
    plt.ylim(Y[:, 1].min() - 10, Y[:, 1].max() + 10)
    plt.show()

def text_classification_part(data_set, w2v_model):
    data_set = data_set[data_set["genre"] != "Not Available"]
    data_set = data_set[data_set["genre"].notnull()]
    data_set['genre'].value_counts().plot.bar()
    # plt.show()

    genres_to_numbers = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    genres_encoded = genres_to_numbers.fit_transform(data_set["genre"].tolist())

    classes = list(genres_to_numbers.classes_)

    X_train, X_test, y_train, y_test = train_test_split(data_set["lyrics"], genres_encoded, test_size=0.2,
                                                        random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    cleared_songs_train, cleared_y_train = clean_text(X_train, y_train)
    cleared_songs_test, cleared_y_test = clean_text(X_test, y_test)

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=15000)

    # text_classification_bow(cleared_songs_train, cleared_songs_test, vectorizer, cleared_y_train, cleared_y_test,
    #                         classes)
    text_classification_avg_vec(w2v_model, cleared_songs_train, cleared_songs_test, cleared_y_train,cleared_y_test, classes)


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
    dict_gen_songs = dictGenreLyrics(path)
    #forPrint = words_3000_print(words_final3000,dict_gen_songs)
    #words_final_50 = makeList50PerGenre(words_final3000,dict_gen_songs,path)
    #display_2D_results(model,words_final_50)

    print('hi')




if __name__ == "__main__":
    main()