import numpy as np
import torch
d=20
print(torch.arange(d))
self_ind = torch.zeros(d,d).long()
self_ind = self_ind.scatter_(dim=1,index = torch.arange(d).unsqueeze(1).long(), src = torch.ones(d,d).long())
print(self_ind)