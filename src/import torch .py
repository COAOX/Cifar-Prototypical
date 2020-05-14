import torch
l = []
v=torch.ones([3,3])
l.extend(v)
l.extend(torch.ones([10,10]))
print(len(l))
