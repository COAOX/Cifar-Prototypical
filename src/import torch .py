import numpy as np
import torch
from torch.nn import functional as F

a = torch.arange(5)
m=5
n=2
c = a.unsqueeze(1).expand(m,n)
b = c.contiguous().view(-1)
print(b.size())
print(c.size())
