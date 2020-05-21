import numpy as np
import torch
from torch.nn import functional as F
def com_proto(img_input):
    #input size = n_class x len(image) x d
    n_class = img_input.size(0)
    n = img_input.size(1)
    d = img_input.size(2)
    ori_prototypes = img_input.mean(1)

    dis_factor = F.softmax(torch.pow((ori_prototypes.expand(n_class,n,d)-img_input),2).sum(2),dim=1)#size = n
    print(dis_factor.size())
    prototypes = img_input.mul(dis_factor.unsqueeze(2).expand(n_class,n,d)).sum(1)
    return prototypes #n_class x d

a = torch.randn(10,10,20)
print(com_proto(a))