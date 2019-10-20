import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(300, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
