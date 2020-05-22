# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.autograd import Variable
from parser_util import get_parser

class BiasLayer(Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        opt = get_parser().parse_args()

        self.alpha = nn.Parameter(torch.ones((1,opt.total_cls), requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros((1,opt.total_cls), requires_grad=True, device="cuda"))
    def forward(self, x):
        x = x.to('cuda')
        return self.alpha[0][:x.size(1)].mul(x) + self.beta[0][:x.size(1)]
    def printParam(self, i):
        print(i, self.alpha, self.beta)

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support
    def forward(self, input, target, opt, old_prototypes, inc_i):
        return prototypical_loss(input, target, self.n_support, opt, old_prototypes, inc_i)


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


def prototypical_loss(input, target, opt, old_prototypes, inc_i,biasLayer,t_prototypes=None):
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
        return target_cpu.eq(c).nonzero()[:opt.num_support_tr].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    cn = opt.class_per_stage
    if inc_i is None:
        classes = target_cpu.unique()
    else:
        classes = torch.arange(inc_i*cn,(inc_i+1)*cn)
    #print(target_cpu.unique())
    n_target = len(target_cpu)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    #n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))
    #if not old_prototypes is None:
    #    print(old_prototypes.size()[0])
    #print((inc_i+1)*opt.class_per_stage)
    n_prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    n_prototypes = n_prototypes.where(n_prototypes==n_prototypes,torch.full(n_prototypes.size(),opt.edge))
    #prototypes = torch.cat([old_prototypes,n_prototypes.clone()],dim=0)

    if old_prototypes is None:
        prototypes = n_prototypes
    elif inc_i is None:
        prototypes = old_prototypes
    elif old_prototypes.size()[0]>=(inc_i+1)*opt.class_per_stage:
        prototypes = torch.cat([old_prototypes[:inc_i*opt.class_per_stage],n_prototypes],dim=0)
    elif not old_prototypes is None:
        prototypes = torch.cat([old_prototypes,n_prototypes],dim=0)
    else:
        prototypes = n_prototypes

    #print("loss prototypes:{}".format(prototypes))

    n_classes = prototypes.size()[0]

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
    dists = biasLayer(dists).to('cpu')
    #print(F.log_softmax(-dists, dim=1).size())
    log_p_y = F.log_softmax(-dists, dim=1)
    softmax_dist = F.softmax(-dists,dim=1)
    #target_inds = torch.arange(0, n_query)
    #target_inds = target_inds.view(1, n_query)
    #target_inds = target_inds.expand(n_classes, n_query).long()
    #target_inds = target_inds.eq()
    #print(dists)
    #prototype_dist = euclidean_dist(prototypes,prototypes)
    
    #print(prototype_dist)
    _, y_hat = log_p_y.max(1)
    #print(prototypes)
    #print(prototype_dist)
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
    target_ninds = target_inds.eq(0)
    c_dist_loss = torch.masked_select(softmax_dist, target_ninds.bool()).mean()
    #proto_dist_mask = prototype_dist.eq(0)
    #proto_dist_mask = proto_dist_mask.eq(0)
    #dist_loss = torch.rsqrt(torch.masked_select(prototype_dist,proto_dist_mask.bool())).mean()
    #print(dist_loss)
     #+log_p_y.squeeze().view(-1).mean()
    if opt.lossF=='NCM':
        loss_val = c_dist_loss-torch.masked_select(log_p_y,target_inds.bool()).mean()
    else:
        entropy = nn.CrossEntropyLoss()
        loss_val= c_dist_loss+entropy(F.softmax(-dists,dim=1),target_cpu)
    #print(log_p_y.size())
    #print(log_p_y)
    if not t_prototypes is None:
        self_dist = euclidean_dist(n_prototypes,t_prototypes)
        d = self_dist.size(0)
        self_ind = torch.zeros(d,d).long()
        self_ind = self_ind.scatter_(dim=1,index = torch.arange(d).unsqueeze(0).long(), src = torch.ones(d,d).long())
        self_dist_loss = torch.masked_select(F.softmax(self_dist,dim=1),self_ind.bool()).mean()
        loss_val = loss_val+self_dist_loss
    #print(y_hat)
    #print(target_inds)
    #loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
    acc_val = y_hat.eq(target_cpu.squeeze()).float().mean()

    return loss_val,  acc_val, n_prototypes


def com_proto(img_input):
    #input size = n_class x len(image) x d
    return img_input.mean(1)
    n_class = img_input.size(0)
    n = img_input.size(1)
    d = img_input.size(2)
    ori_prototypes = img_input.mean(1)
    dis_factor = F.softmax(torch.rsqrt(torch.pow((ori_prototypes.unsqueeze(1).expand(n_class,n,d)-img_input),2).sum(2)),dim=1)#size = n

    prototypes = img_input.mul(dis_factor.unsqueeze(2).expand(n_class,n,d)).sum(1)
    prototypes = torch.where(prototypes==0,torch.full_like(prototypes, 0.0001),prototypes)
    return prototypes #n_class x d
