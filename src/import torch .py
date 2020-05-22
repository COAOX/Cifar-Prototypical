import numpy as np
import torch
from torch.nn import functional as F

a = torch.arange(5)
c = a.unsqueeze(1).expand(5,2)
b = c.contiguous().view(-1)
print(b.size())
print(c.size())
