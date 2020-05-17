# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from model import PreResNet, BiasLayer
from parser_util import get_parser
from cifar import Cifar100
from torch.utils.data import DataLoader
from torchvision import transforms
from exemplar import Exemplar
from dataset import BatchData
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from tqdm import tqdm
import numpy as np
import torch
import os
import argparse
import copy



def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    total_cls = opt.total_cls
    exemplar = Exemplar(max_size, total_cls)
    dataset = Cifar100()

    #dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    #n_classes = len(np.unique(dataset.y))
    #if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        #raise(Exception('There are not enough classes in the dataset in order ' +
         #               'to satisfy the chosen classes_per_it. Decrease the ' +
          #              'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def testf(opt, test_dataloader, model, prototypes, n_per_stage):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    tem_acc = list()
    ind = 0
    count=0
    stage_acc = list()
    for epoch in range(10):
        
        for i, (x, y) in enumerate(tqdm(test_dataloader)):
            t = y.squeeze(-1).size(0)
            
            x, y = x.to(device), y.squeeze(-1).to(device)
            model_output = model(x)
            _, acc= loss_fn(model_output, target=y,
                             n_support=opt.num_support_val, opt=opt, old_prototypes=prototypes,inc_i=None)
            avg_acc.append(acc.item())
            if epoch ==9:
                tem_acc.append(acc.item())
                count = count+t
                if ind<len(n_per_stage) and count>=n_per_stage[ind]:
                #print("ind:{}".format(ind))
                    stage_acc.append(np.mean(tem_acc))
                    tem_acc.clear()
                    ind = ind+1
                    count=0

    avg_acc = np.mean(avg_acc)
    print('Stage Acc: {}'.format(stage_acc))
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc



def compute_NCM_img_id(input,target,n_support, num_support_NCM):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    classes = torch.unique(target_cpu)
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    def class_img(c):
        return target_cpu.eq(c).nonzero().squeeze(1)
    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    NCM = torch.zeros([1,num_support_NCM]).long()
    #print(classes)
    for i,c in enumerate(classes):
        c_img = class_img(c)
        d = input_cpu.size(1)
        n = len(c_img)
        dis = torch.pow(input_cpu[c_img]-prototypes[i].expand(n,d),2).sum(1)
        #print(dis)
        ord_dis,index = torch.sort(dis,dim=0,descending=False)
        #print(ord_dis)
        #print(index)
        img_index = c_img[index]
        #print(c_img)
        #print(img_index)
        #print(img_index)
        NCM = torch.cat([NCM,img_index[:num_support_NCM].unsqueeze(0)],dim=0)
        #print("NCM:{}".format(NCM.size()))
    #print(NCM)
    #print(NCM.view(-1).squeeze())
    return NCM[1:]

def get_mem_tr(support_imgs,num_support_NCM):
    if support_imgs is None:
        return [],[]
    mem_img = torch.split(support_imgs,num_support_NCM,dim=0) # n_class x n_support_NCM x img.size
    n_c = len(mem_img)
    mem_xs=[]
    mem_ys=[]
    for i in range(n_c):
        for m in mem_img[i]:
            mem_xs.append(np.rollaxis(m.squeeze().numpy(),0,3))
        mem_ys.extend([i]*num_support_NCM)
    return mem_xs,mem_ys


def train(opt, model, optim, lr_scheduler):
    '''
    Train the model with the prototypical learning algorithm
    '''
    
    input_transform= Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32,padding=4),
                    ToTensor(),
                    Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

    input_transform_eval= Compose([
                        ToTensor(),
                        Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    total_cls = opt.total_cls
    #exemplar = Exemplar(opt.max_size, total_cls)
    dataset = Cifar100()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')


    test_xs = []
    test_ys = []
    train_xs = []
    train_ys = []
    test_accs = []
    n_per_stage = []
    support_imgs = None
    prototypes = None
    for inc_i in range(opt.stage):
        #exemplar.clear()
        print(f"Incremental num : {inc_i}")
        train, val, test = dataset.getNextClasses(inc_i)
        train_x, train_y = zip(*train)
        val_x, val_y = zip(*val)
        test_x, test_y = zip(*test)
        #print(f"train:{train_y}")
        train_y_hot = dense_to_one_hot(train_y,100)
        val_y = dense_to_one_hot(val_y,100)
        test_y = dense_to_one_hot(test_y,100)
        test_xs.extend(test_x)
        test_ys.extend(test_y)
        train_xs.clear()
        train_ys.clear()
        #print(f"train_y:{train_y} ,val_y:{val_y}, test_y:{test_y}")
        #train_xs, train_ys = exemplar.get_exemplar_train()
        train_xs.extend(train_x)
        train_xs.extend(val_x)
        train_ys.extend(train_y)
        train_ys.extend(val_y)
        train_xNCM = train_xs[:]
        train_yNCM = train_ys[:]
        NCM_dataloader = DataLoader(BatchData(train_xNCM, train_yNCM, input_transform),
                    batch_size=opt.NCM_batch, shuffle=True, drop_last=True)
        mem_xs,mem_ys = get_mem_tr(support_imgs,opt.num_support_NCM)

        train_xs.extend(mem_xs)
        train_ys.extend(mem_ys)
        tr_dataloader = DataLoader(BatchData(train_xs, train_ys, input_transform),
                    batch_size=opt.batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(BatchData(val_x, val_y, input_transform_eval),
                    batch_size=opt.batch_size, shuffle=False)
        test_data = DataLoader(BatchData(test_xs, test_ys, input_transform_eval),
                    batch_size=opt.batch_size, shuffle=False)
        #exemplar.update(total_cls//opt.stage, (train_x, train_y), (val_x, val_y))
        n_per_stage.append(len(test_data))
        for epoch in range(opt.epochs):
            print('=== Epoch: {} ==='.format(epoch))
            #tr_iter = iter(tr_dataloader)
            model.train()
            train_acc.clear()
            train_loss.clear()
            #optim.zero_grad()
            
            for i, (cx, cy) in enumerate(tqdm(tr_dataloader)):
                optim.zero_grad()
                #print("x:{},y:{}".format(x.size(),y.squeeze().size()))
                x, y = cx.to(device), cy.squeeze().to(device)

                model_output = model(x)
                #print(model_output.size())
                #print("#######model_output:{}".format(model_output.size()))



                loss, acc= loss_fn(model_output, target=y, n_support=opt.num_support_tr, opt=opt, old_prototypes=None if prototypes is None else prototypes.detach(), inc_i=inc_i)

                loss.backward(retain_graph=True)
                optim.step()
                train_loss.append(loss.item())
                train_acc.append(acc.item())
            if epoch == opt.epochs-1:
                for x,y in NCM_dataloader:
                    cx,y = x.to(device),y.squeeze().to(device)
                    model_output = model(cx)

                    print("Compute NCM")
                    NCM_img_id = compute_NCM_img_id(model_output,y,opt.num_support_tr,opt.num_support_NCM)#num_support_NCM*stage_per_classes
                    support_img = x.index_select(0,NCM_img_id.view(-1).squeeze())#num_support_NCM*stage_per_classes,img_size
                    break
            avg_loss = np.mean(train_loss)
            avg_acc = np.mean(train_acc)
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
            lr_scheduler.step()

            #if val_dataloader is None:
                #continue
            model.eval()
            val_acc.clear()
            val_loss.clear()
            for i, (x, y) in enumerate(tqdm(val_dataloader)):
                x, y = x.to(device), y.squeeze().to(device)
                model_output = model(x)
                loss, acc= loss_fn(model_output, target=y, n_support=opt.num_support_val, opt=opt, old_prototypes=None if prototypes is None else prototypes.detach(),inc_i=inc_i)
                val_loss.append(loss.item())
                val_acc.append(acc.item())
            avg_loss = np.mean(val_loss)
            avg_acc = np.mean(val_acc)
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
                best_acc)
            print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                avg_loss, avg_acc, postfix))
            if avg_acc >= best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = model.state_dict()
        
        #pp = torch.ones([20,256])
        if inc_i ==0:
            support_imgs = support_img
        else:
            #prototypes = torch.ones([20,256])
            #tem = torch.split(support_img,opt.n_support,dim=0)
            support_imgs = torch.cat([support_imgs,support_img],dim=0)#n_classes x n_support x img.size
        
        if not support_imgs is None:
            prototypes = torch.stack(torch.split(model(support_imgs.to(device)),opt.num_support_NCM,dim=0))#n_class x n_support x prototypes.size()--256
            #print(prototypes)
            #print(prototypes.size())
            prototypes = prototypes.mean(1).to('cpu')
        print('Testing with last model..')
        testf(opt=opt, test_dataloader=test_data, model=model, prototypes=prototypes.to('cpu'), n_per_stage=n_per_stage)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    #testf(opt=opt, test_dataloader=test_data, model=model, prototypes=prototypes)
    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                   name + '.txt'), locals()[name])




def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    testf(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)


    #tr_dataloader = init_dataloader(options, 'train')
    #val_dataloader = init_dataloader(options, 'val')
    # trainval_dataloader = init_dataloader(options, 'trainval')
    #test_dataloader = init_dataloader(options, 'test')
    model = PreResNet(32,options.total_cls).cuda()
    #model = init_protonet(options)
    model = nn.DataParallel(self.model, device_ids=[0])
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    train(opt=options,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)

    print("----------train finished----------")
    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    #label_dense = torch.LongTensor(np.array([labels_dense],dtype=float).T)
    #label_dense = torch.LongTensor(len(labels_dense),1).random_()%num_classes
    #label_dense = label_dense.transpose(1,0)
    #print(f"label_dense: {label_dense}")
    #y_onehot = torch.FloatTensor(len(label_dense), num_classes)
    #y_onehot.zero_()
    #y_onehot.scatter_(1, label_dense, 1)
    #print(f"onehot: {y_onehot}")
    return labels_dense

if __name__ == '__main__':
    main()
