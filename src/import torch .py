import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

a = torch.randn(5,64,16,16)
def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
b = conv_block(64,64)
c = b(a)
print(c.size())
