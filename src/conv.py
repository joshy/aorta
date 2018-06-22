import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch = nn.BatchNorm2d()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.out = nn.Conv2d(32, 64, 1)


    def forward(self, x):
        bn = self.batch(x)
        c1 = self.conv1(bn)
        c2 = self.conv2(c1)
        out = self.out(c2)
        return F.sigmoid(out)


net = Conv()