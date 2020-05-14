# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
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

def testf(opt, test_dataloader, model, prototypes):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        for i, (x, y) in enumerate(tqdm(test_dataloader)):
            x, y = x.to(device), y.squeeze(-1).to(device)
            model_output = model(x)
            _, acc, _ = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val, opt=opt, old_prototypes=prototypes,inc_i=None)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc

def train(opt, model, optim, lr_scheduler):
    '''
    Train the model with the prototypical learning algorithm
    '''
    torch.autograd.set_detect_anomaly(True)
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
    exemplar = Exemplar(opt.max_size, total_cls)
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
    prototypes = None

    for inc_i in range(opt.stage):
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

        #print(f"train_y:{train_y} ,val_y:{val_y}, test_y:{test_y}")
        train_xs, train_ys = exemplar.get_exemplar_train()
        train_xs.extend(train_x)
        train_xs.extend(val_x)
        train_ys.extend(train_y)
        train_ys.extend(val_y)
        print(len(train_xs))
        print(len(test_xs))
        tr_dataloader = DataLoader(BatchData(train_xs, train_ys, input_transform),
                    batch_size=opt.batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(BatchData(val_x, val_y, input_transform_eval),
                    batch_size=opt.batch_size, shuffle=False)
        test_data = DataLoader(BatchData(test_xs, test_ys, input_transform_eval),
                    batch_size=opt.batch_size, shuffle=False)


        for epoch in range(opt.epochs):
            print('=== Epoch: {} ==='.format(epoch))
            #tr_iter = iter(tr_dataloader)
            model.train()
            print("######train#######")
            train_acc.clear()
            train_loss.clear()
            for i, (x, y) in enumerate(tqdm(tr_dataloader)):
                optim.zero_grad()
                #print("x:{},y:{}".format(x.size(),y.squeeze().size()))
                x, y = x.to(device), y.squeeze().to(device)

                model_output = model(x)
                #print(model_output.size())
                #print("#######model_output:{}".format(model_output.size()))
                loss, acc, prototype = loss_fn(model_output, target=y, n_support=opt.num_support_tr, opt=opt, old_prototypes=prototypes,inc_i=inc_i)
                
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                train_acc.append(acc.item())

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
                loss, acc, prototype = loss_fn(model_output, target=y,
                                    n_support=opt.num_support_val, opt=opt, old_prototypes=prototypes,inc_i=inc_i)
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
        if not prototypes is None:
            prototypes = torch.cat([prototypes,copy.deepcopy(prototype)],dim=0)
        else:
            prototypes = prototype

        print('Testing with last model..')
        testf(opt=opt,
            test_dataloader=test_data,
             model=model, prototypes=prototypes)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    testf(opt=opt,
         test_dataloader=test_data,
         model=model, prototypes=prototypes)
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

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    train(opt=options,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)


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
