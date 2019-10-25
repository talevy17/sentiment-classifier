from torch import optim

import torch.nn as nn
import numpy
import torch
import random


class SimpleNeuralNet:
    def __init__(self, model):
        self.word2vec = model

    def shuffle(self, x, y):
        zip_x_y = list(zip(x, y))
        random.shuffle(zip_x_y)
        new_x, new_y = zip(*zip_x_y)
        return new_x, new_y

    def train(self, data, label_set,network):
        self.opt = torch.optim.Adam(network.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        labels = numpy.asarray(label_set)
        for epoch in range(5):
            network.train()
            # print('Epoch: ' + str(epoch))
            data, labels = self.shuffle(data, labels)
            for k, (word, label) in enumerate(zip(data, labels)):
                temp = torch.from_numpy(self.word2vec[word])
                temp = temp.reshape(-1, temp.size(0))
                prediction = network(temp)
                loss = loss_function(prediction, torch.tensor(label))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            # validation(conv, valid_set, device)

    def validate(self, data, labels,network):
        network.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for word, label in zip(data, labels):
                # calc on device (CPU or GPU)
                # calc the prediction vector using the model's forward pass.
                temp = torch.from_numpy(self.word2vec[word])
                temp = temp.reshape(-1, temp.size(0))
                pred = network(temp)
                total += 1
                if pred.data * torch.tensor(label).data > 0:
                    correct += 1
            # print the accuracy of the model.
            print('Test Accuracy of the model: {}%'.format((correct / total) * 100))

    def test(self, test, model_test,network):
        dict = {}
        print(model_test.wv.vocab)
        network.eval()
        with torch.no_grad():
            for word in test:
                # calc on device (CPU or GPU)
                # calc the prediction vector using the model's forward pass.
                if word in model_test.wv.vocab:
                    temp = torch.from_numpy(model_test[word])
                    temp = temp.reshape(-1, temp.size(0))
                    pred = (network(temp))
                    pred = pred.data[0].item()
                    dict[word] = pred
                    #print(word + ' , ' + str(pred))
                    #print(str(word) + ' ' + str(pred.data[0]))
        return dict


