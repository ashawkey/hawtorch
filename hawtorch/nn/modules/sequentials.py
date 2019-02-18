import torch
import torch.nn as nn
from .normalization import GroupNorm

class fcdr(nn.Module):
    def __init__(self, in_features, out_features, p=0.5, activation=True):
        super(fcdr, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(p=p),
        )
        if activation: self.seq.add_module("activatoin", nn.ReLU(inplace=True))
    def forward(self, x):
        return self.seq(x)

class conv2dbr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(conv2dbr, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        if activation: self.seq.add_module("activation", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)
    
class conv2dgr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(conv2dgr, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            GroupNorm(out_channels),
        )
        if activation: self.seq.add_module("activation", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)

class resnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(resnetBlock, self).__init__()
        self.conv1 = conv2dbr(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.conv2 = conv2dbr(out_channels, out_channels, kernel_size, stride, padding=padding, activation=False)
        if stride != 1:
            self.downsample = conv2dbr(in_channels, out_channels, stride, activation=False)
        else:
            self.downsample=None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)

        return x
