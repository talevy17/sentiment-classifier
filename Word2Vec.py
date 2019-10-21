from Tokenizer import Tokenizer as tk
from DataCleaner import DataCleaner as dc
from SimpleNeuralNet import SimpleNeuralNet
from enum import Enum
import numpy
import torch
from torch import nn
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


def main():
    file = open(Constants.SEM_EVAL.value, 'r')
    data = shuffle_data(file.readlines())
    file.close()
    data, labels, tok = split_data(data)
    model = gensim.models.Word2Vec(tok, min_count=1, size=300, window=5, sg=1, iter=1)
    # token = tk('./Dataset/lyrics15LIN.csv', ['english', 'spanish'], '''!()-[]{};:"\,<>./?@#$%^&*_~''')
    toke = dc('./Dataset/lyrics15LIN.csv')
    token = toke.clean_data()
    token = toke.tokenize(token)
    print(token)
    model_test = gensim.models.Word2Vec(min_count=1, size=300, window=5, sg=1, iter=1)
    #model_test.build_vocab(token)
    #model_test.train(token,total_examples=model.corpus_count,epochs=1)
    # train_data, train_labels, valid_data, valid_labels = cross_validation(data, labels)
    fc = SimpleNeuralNet(model)
    # print(len(train_labels))
    fc.train(data, labels)
    fc.test(tok, model_test)
    # fc.validate(valid_data, valid_labels)
    # net.train(data, labels)
    # net.validate(data, labels)


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