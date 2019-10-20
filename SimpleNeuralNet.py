from FullyConnected import FullyConnected as Fc
import torch.nn as nn
import numpy
import torch


class SimpleNeuralNet:
    def __init__(self, model):
        self.word2vec = model
        self.net = Fc()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.01)

    def train(self, data, label_set):
        loss_function = nn.MSELoss()
        labels = numpy.asarray(label_set)
        for epoch in range(1):
            self.net.train()
            # print('Epoch: ' + str(epoch))
            for k, (word, label) in enumerate(zip(data, labels)):
                temp = torch.from_numpy(self.word2vec[word])
                temp = temp.reshape(-1, temp.size(0))
                prediction = self.net(temp)
                loss = loss_function(prediction, torch.tensor(label))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            # validation(conv, valid_set, device)

    def validate(self, data, labels):
        self.net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for word, label in zip(data, labels):
                # calc on device (CPU or GPU)
                # calc the prediction vector using the model's forward pass.
                temp = torch.from_numpy(self.word2vec[word])
                temp = temp.reshape(-1, temp.size(0))
                pred = self.net(temp)
                print(pred)
                total += 1
                if pred.data * torch.tensor(label).data > 0:
                    correct += 1
            # print the accuracy of the model.
            print('Test Accuracy of the model: {}%'.format((correct / total) * 100))

    def test(self, test):
        self.net.eval()
        with torch.no_grad():
            for word in test:
                # calc on device (CPU or GPU)
                # calc the prediction vector using the model's forward pass.
                pred = self.net(self.word2vec[word])
                print(word+' '+pred)
