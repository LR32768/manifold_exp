from typing import Any, List, Type, Union, Optional

import torch
from torch import Tensor
from torch import nn


class MLP(nn.Module):

    def __init__(
            self,
            in_channels = 3*224*224,
            hid_channels = 5,
            num_classes = 2,
            use_bn = False
            ):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.use_bn = use_bn

        self.fc1 = nn.Linear(in_channels, hid_channels, bias=True)
        if use_bn:
            self.bn = nn.BatchNorm1d(hid_channels)
        self.relu = nn.LeakyReLU(0.1, True)
        self.fc2 = nn.Linear(hid_channels, num_classes, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def MLP_1d(use_bn=False):
    return MLP(hid_channels=1, use_bn=use_bn)

def MLP_2d(use_bn=False):
    return MLP(hid_channels=2, use_bn=use_bn)

def MLP_5d(use_bn=False):
    return MLP(hid_channels=5, use_bn=use_bn)

def MLP_10d(use_bn=False):
    return MLP(hid_channels=10, use_bn=use_bn)

def MLP_32d(use_bn=False):
    return MLP(hid_channels=32, use_bn=use_bn)

def MLP_128d(use_bn=False):
    return MLP(hid_channels=128, use_bn=use_bn)

def MLP_128d_bn(use_bn=True):
    return MLP(hid_channels=128, use_bn=use_bn)

def MLP_256d(use_bn=False):
    return MLP(hid_channels=128, use_bn=use_bn)

def MLP_256d_bn(use_bn=True):
    return MLP(hid_channels=128, use_bn=use_bn)

if __name__ == "__main__":
    model = MLP_1d()
    inp = torch.randn(1, 3, 224, 224)
    print(model(inp))