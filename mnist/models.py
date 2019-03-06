import torch
import torch.nn as nn
import torch.optim as optim

import hawtorch
import hawtorch.nn as hnn

class MNISTClassifier(nn.Module):
    __name__ = "MNISTClassifier"
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv0 = hnn.conv2dbr(1, 32, 5, 1)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.conv1 = hnn.conv2dbr(32, 64, 5, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc0 = hnn.fcdr(64*4*4, 512)
        self.fc1 = hnn.fcdr(512, 10, activation=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(-1, 64*4*4)
        x = self.fc0(x)
        x = self.fc1(x)
        
        return x
