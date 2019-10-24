import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc1 = (nn.Linear(300, 100))
        self.fc2 = (nn.Linear(100, 1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        return x
