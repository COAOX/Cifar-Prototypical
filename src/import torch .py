import torch
v=torch.Tensor([[1],[2],[3]])
n=v.size(0)
one_hot = torch.zeros(n,10).long()
print(one_hot.scatter_(dim=1, index=v.long(), src=torch.ones(n, 10).long()))
