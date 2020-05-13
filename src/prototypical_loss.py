# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.autograd import Variable

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N_query(n) x D
    # y: N_Classes(m) x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    #print("n:{},m:{},d:{},ysize:{}".format(n,m,d,y.size(1)))
    if d != y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support, opt, old_prototypes, inc_i):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    
    n_target = len(target_cpu)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    #n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    #if not old_prototypes is None:
    #    print(old_prototypes.size()[0])
    #print((inc_i+1)*opt.class_per_stage)
    n_prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    if old_prototypes is None:
        prototypes = n_prototypes
    elif inc_i is None or old_prototypes.size()[0]>=(inc_i+1)*opt.class_per_stage:
        prototypes = old_prototypes
    elif not old_prototypes is None:
        prototypes = torch.cat([old_prototypes,n_prototypes],dim=0)
    else:
        prototypes = n_prototypes
    #print(old_prototypes)
    #print("loss prototypes:{}".format(prototypes))
    print(prototypes.size())
    n_classes = prototypes.size()[0]
    #print(n_classes)
    # FIXME when torch will support where as np
    #print(n_support)
    #print(target_cpu)
    #for x in classes:
        #print("{}:{}".format(x,target_cpu.eq(x).nonzero()))
    #print(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes)))
    #query_idlist = list(map(lambda c: target_cpu.eq(c).nonzero(), classes))
    #query_idxs = torch.cat(query_idlist).view(-1)
    #print(query_idxs)
    #query_samples = torch.stack([input_cpu[query_lists] for query_lists in query_idxs])
    #query_samples = input_cpu[query_idxs]
    #print(query_samples.size())
    #print(prototypes.size())
    n_query = len(input_cpu)
    dists = euclidean_dist(input_cpu, prototypes)
    #print(F.log_softmax(-dists, dim=1).size())
    log_p_y = F.log_softmax(-dists, dim=1)
    #print(log_p_y)
    #target_inds = torch.arange(0, n_query)
    #target_inds = target_inds.view(1, n_query)
    #target_inds = target_inds.expand(n_classes, n_query).long()
    #target_inds = target_inds.eq()
    #print(dists)
    _, y_hat = log_p_y.max(1)
    #print(y_hat)
    #print(target_cpu)
    #print( y_hat.eq(target_cpu.squeeze()).float().mean())
    #target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()
    #target_inds = Variable(target_inds, requires_grad=False)
    target_inds = torch.zeros(len(target_cpu),n_classes).long()
    
    target_inds = target_inds.scatter_(dim=1, index=target_cpu.unsqueeze(1).long(), src=torch.ones(len(target_cpu), n_classes).long())
    #target_inds = target_inds.transpose(0,1)
    #print(target_inds.size())
    #print(log_p_y.type())
    #target_inds = [target_inds.index_put_(query_idl,query_idl) for query_idl in query_idlist]

    loss_val = -torch.masked_select(log_p_y,target_inds.bool()).sum() #+log_p_y.squeeze().view(-1).mean()
    #print(log_p_y.size())
    #print(log_p_y)

    #print(y_hat)
    #print(target_inds)
    #loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
    acc_val = y_hat.eq(target_cpu.squeeze()).float().mean()

    return loss_val,  acc_val, n_prototypes
